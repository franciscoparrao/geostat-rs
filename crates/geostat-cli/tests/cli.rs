//! End-to-end CLI tests driving the built `geostat` binary directly
//! (`std::process::Command`, no extra test-only dependency): variogram fit
//! -> kriging -> cross-validation on a small synthetic fixture, plus a few
//! error-path checks (AUDIT-2026-07-v2.md §6 Fase 5 — the CLI crate had no
//! integration tests at all before this file).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique path under the OS temp dir (pid + counter avoids collisions
/// across parallel `#[test]` threads without a `tempfile` dependency).
fn temp_path(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "geostat-cli-test-{}-{n}-{name}",
        std::process::id()
    ))
}

fn geostat() -> Command {
    Command::new(env!("CARGO_BIN_EXE_geostat"))
}

fn run(cmd: &mut Command) -> Output {
    cmd.output().expect("failed to spawn geostat binary")
}

fn assert_success(out: &Output, context: &str) {
    assert!(
        out.status.success(),
        "{context} failed (status {:?})\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A small synthetic point set with a genuine spatial structure (not pure
/// noise), so `--fit best` converges to a sensible model and kriging beats
/// the mean.
fn write_fixture(path: &Path) {
    let mut csv = String::from("x,y,z\n");
    let mut seed: u64 = 12345;
    let mut next = || {
        // xorshift64: no rand dependency needed for a deterministic fixture.
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64 / u64::MAX as f64) * 100.0
    };
    for _ in 0..80 {
        let x = next();
        let y = next();
        let z = (x / 15.0).sin() + (y / 20.0).cos() + 5.0;
        csv.push_str(&format!("{x},{y},{z}\n"));
    }
    std::fs::write(path, csv).unwrap();
}

#[test]
fn variogram_krige_cv_pipeline_end_to_end() {
    let data = temp_path("fixture.csv");
    let model = temp_path("model.json");
    let vario_out = temp_path("vario.csv");
    let kriged = temp_path("kriged.csv");
    write_fixture(&data);

    let out = run(geostat()
        .args(["variogram", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "--fit", "best", "--model-out"])
        .arg(&model)
        .arg("-o")
        .arg(&vario_out));
    assert_success(&out, "variogram");
    assert!(model.exists(), "model JSON was not written");
    assert!(vario_out.exists(), "variogram bins CSV was not written");

    let model_json = std::fs::read_to_string(&model).unwrap();
    assert!(
        model_json.contains("\"nugget\""),
        "model JSON missing expected shape: {model_json}"
    );

    let out = run(geostat()
        .args(["krige", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "-m"])
        .arg(&model)
        .args(["--nx", "10", "--ny", "10", "-o"])
        .arg(&kriged));
    assert_success(&out, "krige");
    let kriged_csv = std::fs::read_to_string(&kriged).unwrap();
    let n_lines = kriged_csv.lines().count();
    assert_eq!(n_lines, 1 + 100, "expected a header plus 10x10 grid rows");
    assert_eq!(
        kriged_csv.lines().next().unwrap(),
        "x,y,prediction,variance"
    );

    let out = run(geostat()
        .args(["cv", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "-m"])
        .arg(&model));
    assert_success(&out, "cv");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Leave-one-out cross-validation"),
        "unexpected cv output: {stdout}"
    );
    assert!(stdout.contains("RMSE"));

    for p in [&data, &model, &vario_out, &kriged] {
        std::fs::remove_file(p).ok();
    }
}

#[test]
fn block_cv_and_kfold_cv_print_their_own_method_label() {
    // Regression guard: `print_cv_report` used to always print "Leave-one-out
    // cross-validation" regardless of --blocks/--folds (AUDIT-2026-07-v2.md
    // §6 Fase 5 fix, found while writing the README's block-CV example).
    let data = temp_path("fixture_cv.csv");
    let model = temp_path("model_cv.json");
    write_fixture(&data);
    let out = run(geostat()
        .args(["variogram", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "--fit", "best", "--model-out"])
        .arg(&model));
    assert_success(&out, "variogram (for cv fixture)");

    let out = run(geostat()
        .args(["cv", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "-m"])
        .arg(&model)
        .args(["--blocks", "3,3"]));
    assert_success(&out, "cv --blocks");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Spatial block cross-validation"),
        "unexpected cv --blocks output: {stdout}"
    );

    let out = run(geostat()
        .args(["cv", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "-m"])
        .arg(&model)
        .args(["--folds", "4"]));
    assert_success(&out, "cv --folds");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("K-fold cross-validation"),
        "unexpected cv --folds output: {stdout}"
    );

    for p in [&data, &model] {
        std::fs::remove_file(p).ok();
    }
}

#[test]
fn krige_rejects_targets_in_plain_2d_mode_and_raster_with_a_non_gpkg_output() {
    // AUDIT-2026-07-v3.md §1.14: plain 2-D `krige` used to silently ignore
    // `--targets` (kriging the default grid instead of the requested
    // points) and `--raster` with a non-.gpkg output (writing a point CSV
    // instead of the requested raster), with no indication either flag was
    // dropped.
    let data = temp_path("fixture_targets_raster.csv");
    let model = temp_path("model_targets_raster.json");
    let targets = temp_path("targets.csv");
    write_fixture(&data);
    std::fs::write(&targets, "x,y\n10.0,10.0\n20.0,20.0\n").unwrap();

    let out = run(geostat()
        .args(["variogram", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "--fit", "best", "--model-out"])
        .arg(&model));
    assert_success(&out, "variogram (for targets/raster fixture)");

    let out = run(geostat()
        .args(["krige", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "-m"])
        .arg(&model)
        .args(["--targets"])
        .arg(&targets)
        .args(["-o"])
        .arg(temp_path("should_not_be_written.csv")));
    assert!(
        !out.status.success(),
        "--targets in plain 2-D mode should be rejected, not silently ignored"
    );

    let out = run(geostat()
        .args(["krige", "-i"])
        .arg(&data)
        .args(["--value-col", "z", "-m"])
        .arg(&model)
        .args(["--raster", "--nx", "5", "--ny", "5", "-o"])
        .arg(temp_path("should_not_be_written.csv")));
    assert!(
        !out.status.success(),
        "--raster with a non-.gpkg output should be rejected, not silently dropped"
    );

    for p in [&data, &model, &targets] {
        std::fs::remove_file(p).ok();
    }
}

#[test]
fn robust_estimator_flag_changes_the_reported_gamma() {
    // AUDIT-2026-07-v2.md §4/§7 Fase 6 item #16: robust estimators
    // (Cressie-Hawkins/Dowd/madogram) previously had no CLI surface at all.
    let data = temp_path("fixture_estimator.csv");
    write_fixture(&data);

    let matheron = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "z",
        "--n-lags",
        "5",
    ]));
    assert_success(&matheron, "variogram (matheron)");
    let matheron_out = String::from_utf8_lossy(&matheron.stdout).into_owned();
    assert!(!matheron_out.contains("Estimator:"));

    let dowd = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "z",
        "--n-lags",
        "5",
        "--estimator",
        "dowd",
    ]));
    assert_success(&dowd, "variogram --estimator dowd");
    let dowd_out = String::from_utf8_lossy(&dowd.stdout).into_owned();
    assert!(dowd_out.contains("Estimator: dowd"));
    // Different estimator, same data: the printed gamma values must differ
    // (this is the whole point of the flag actually reaching the engine).
    assert_ne!(matheron_out, dowd_out);

    let bad = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "z",
        "--estimator",
        "bogus",
    ]));
    assert!(!bad.status.success());

    std::fs::remove_file(&data).ok();
}

#[test]
fn madogram_with_fit_is_rejected_but_madogram_alone_and_matheron_fit_still_work() {
    // AUDIT-2026-07-v3.md §1.11: the madogram is on a different (non-
    // quadratic) scale than gamma -- fitting it directly used to silently
    // distort the kriging model's sill/nugget/shape with no warning.
    let data = temp_path("fixture_madogram.csv");
    write_fixture(&data);

    let madogram_fit = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "z",
        "--estimator",
        "madogram",
        "--fit",
        "best",
    ]));
    assert!(
        !madogram_fit.status.success(),
        "--estimator madogram --fit best should be rejected"
    );

    let madogram_alone = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "z",
        "--estimator",
        "madogram",
    ]));
    assert_success(&madogram_alone, "variogram --estimator madogram (no --fit)");

    let matheron_fit = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "z",
        "--fit",
        "spherical",
    ]));
    assert_success(&matheron_fit, "variogram --fit spherical (default estimator)");

    std::fs::remove_file(&data).ok();
}

#[test]
fn coincident_pairs_are_reported_by_the_cli() {
    let data = temp_path("fixture_coincident.csv");
    // Two points share a location; the CLI must surface that instead of
    // silently dropping the pair.
    std::fs::write(
        &data,
        "x,y,z\n0,0,1.0\n0,0,5.0\n1,0,2.0\n2,0,3.0\n3,1,4.0\n",
    )
    .unwrap();

    let out = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "z",
        "--n-lags",
        "4",
    ]));
    assert_success(&out, "variogram (coincident points)");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("coincident"),
        "expected a coincident-pairs note, got: {stdout}"
    );

    std::fs::remove_file(&data).ok();
}

#[test]
fn missing_column_gives_a_clear_error_not_a_panic() {
    let data = temp_path("fixture_badcol.csv");
    std::fs::write(&data, "east,north,val\n0,0,1.0\n1,1,2.0\n").unwrap();

    let out = run(geostat().args(["variogram", "-i"]).arg(&data).args([
        "--value-col",
        "val",
        "--fit",
        "best",
    ]));
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("column 'x' not found"),
        "expected a clear missing-column error, got: {stderr}"
    );

    std::fs::remove_file(&data).ok();
}

#[test]
fn gpkg_input_fails_clearly_instead_of_falling_through_to_the_csv_parser() {
    // AUDIT-2026-07-v2.md §1.9: 3-D/drift/error-column reads on a .gpkg path
    // used to silently fall through to the CSV parser and fail with a
    // confusing "column not found" error over binary garbage. The path need
    // not be a real GeoPackage: the guard fires on the extension alone,
    // before any file is opened.
    let fake_gpkg = temp_path("fake.gpkg");
    let out_csv = temp_path("fake-out.csv");
    let model = temp_path("fake-model.json");
    let targets = temp_path("fake-targets.csv");
    std::fs::write(
        &model,
        r#"{"nugget":0.1,"structures":[{"kind":"spherical","sill":0.9,"range":10.0}]}"#,
    )
    .unwrap();
    std::fs::write(&targets, "x,y,z\n0,0,0\n").unwrap();

    let out = run(geostat().args([
        "krige",
        "-i",
        fake_gpkg.to_str().unwrap(),
        "--value-col",
        "z",
        "--z-col",
        "z",
        "-m",
        model.to_str().unwrap(),
        "--targets",
        targets.to_str().unwrap(),
        "-o",
        out_csv.to_str().unwrap(),
    ]));
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("GeoPackage") || stderr.contains(".gpkg"),
        "expected a clear GeoPackage-not-supported error, got: {stderr}"
    );
    assert!(
        !stderr.contains("available columns"),
        "must not fall through to the CSV parser: {stderr}"
    );

    for p in [&model, &targets] {
        std::fs::remove_file(p).ok();
    }
}
