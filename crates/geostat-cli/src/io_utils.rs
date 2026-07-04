//! CSV input/output helpers for the CLI.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use geostat_core::{
    CvResult, ExperimentalVariogram, Grid2D, KrigingEstimate, Lmc, PointSet, VariogramModel,
};

fn parse_field(record: &csv::StringRecord, idx: usize) -> Option<f64> {
    let s = record.get(idx)?.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("na") || s.eq_ignore_ascii_case("nan") {
        return None;
    }
    s.parse().ok()
}

fn column_indices(headers: &csv::StringRecord, names: &[&str]) -> Result<Vec<usize>> {
    names
        .iter()
        .map(|name| {
            headers.iter().position(|h| h == *name).with_context(|| {
                format!(
                    "column '{name}' not found; available columns: {}",
                    headers.iter().collect::<Vec<_>>().join(", ")
                )
            })
        })
        .collect()
}

/// Reads point data from a CSV file with named columns, plus optional extra
/// numeric columns (e.g. drift covariates or a secondary variable). Rows
/// with missing entries in any requested column are skipped with a warning.
pub fn read_points_with_extras(
    path: &Path,
    x_col: &str,
    y_col: &str,
    value_col: &str,
    extra_cols: &[String],
) -> Result<(PointSet, Vec<Vec<f64>>)> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("cannot open input file {}", path.display()))?;
    let headers = rdr.headers().context("cannot read CSV header")?.clone();
    let mut names: Vec<&str> = vec![x_col, y_col, value_col];
    names.extend(extra_cols.iter().map(String::as_str));
    let idx = column_indices(&headers, &names)?;

    let mut coords = Vec::new();
    let mut values = Vec::new();
    let mut extras = Vec::new();
    let mut skipped = 0usize;
    for (row, record) in rdr.records().enumerate() {
        let record = record.with_context(|| format!("cannot parse CSV row {}", row + 2))?;
        let fields: Vec<Option<f64>> = idx.iter().map(|&i| parse_field(&record, i)).collect();
        if fields.iter().any(Option::is_none) {
            skipped += 1;
            continue;
        }
        let f: Vec<f64> = fields.into_iter().map(Option::unwrap).collect();
        coords.push([f[0], f[1]]);
        values.push(f[2]);
        extras.push(f[3..].to_vec());
    }
    if skipped > 0 {
        eprintln!("warning: skipped {skipped} rows with missing or non-numeric entries");
    }
    if coords.is_empty() {
        bail!("no valid data rows in {}", path.display());
    }
    Ok((PointSet::new(coords, values)?, extras))
}

/// Reads point data (x, y, value) from a CSV file.
pub fn read_points(path: &Path, x_col: &str, y_col: &str, value_col: &str) -> Result<PointSet> {
    Ok(read_points_with_extras(path, x_col, y_col, value_col, &[])?.0)
}

/// Reads 3-D point data (x, y, z, value) from a CSV file.
pub fn read_points3(
    path: &Path,
    x_col: &str,
    y_col: &str,
    z_col: &str,
    value_col: &str,
) -> Result<PointSet<3>> {
    let (flat, extras) =
        read_points_with_extras(path, x_col, y_col, z_col, &[value_col.to_string()])?;
    // The generic reader yields (x, y) coords with z as "value" and the real
    // value as the extra column; reassemble into 3-D points.
    let coords: Vec<[f64; 3]> = flat
        .coords()
        .iter()
        .zip(flat.values())
        .map(|(c, &z)| [c[0], c[1], z])
        .collect();
    Ok(PointSet::new(
        coords,
        extras.iter().map(|r| r[0]).collect(),
    )?)
}

/// Reads 3-D prediction targets (x, y, z) from a CSV file.
pub fn read_targets3(path: &Path, x_col: &str, y_col: &str, z_col: &str) -> Result<Vec<[f64; 3]>> {
    let (coords2, extras) = read_targets(path, x_col, y_col, &[z_col.to_string()])?;
    Ok(coords2
        .iter()
        .zip(&extras)
        .map(|(c, e)| [c[0], c[1], e[0]])
        .collect())
}

/// Writes 3-D per-point predictions as CSV (`x,y,z,prediction,variance`).
pub fn write_estimates3_csv(
    path: &Path,
    coords: &[[f64; 3]],
    estimates: &[KrigingEstimate],
) -> Result<()> {
    let mut w = writer(path)?;
    writeln!(w, "x,y,z,prediction,variance")?;
    for (c, e) in coords.iter().zip(estimates) {
        writeln!(w, "{},{},{},{},{}", c[0], c[1], c[2], e.value, e.variance)?;
    }
    Ok(())
}

/// Writes indicator-kriging ccdfs as CSV
/// (`x,y,F_1..F_K,e_type,cond_var`).
pub fn write_ik_csv(
    path: &Path,
    coords: &[[f64; 2]],
    estimates: &[geostat_core::CcdfEstimate],
    n_cutoffs: usize,
) -> Result<()> {
    let mut w = writer(path)?;
    let fs: Vec<String> = (1..=n_cutoffs).map(|k| format!("F{k}")).collect();
    writeln!(w, "x,y,{},e_type,cond_var", fs.join(","))?;
    for (c, e) in coords.iter().zip(estimates) {
        let fs: Vec<String> = e.ccdf.iter().map(|f| f.to_string()).collect();
        writeln!(
            w,
            "{},{},{},{},{}",
            c[0],
            c[1],
            fs.join(","),
            e.e_type,
            e.cond_var
        )?;
    }
    Ok(())
}

/// Target coordinates plus their covariate rows.
pub type Targets = (Vec<[f64; 2]>, Vec<Vec<f64>>);

/// Reads prediction targets (x, y plus covariate columns) from a CSV file.
pub fn read_targets(
    path: &Path,
    x_col: &str,
    y_col: &str,
    extra_cols: &[String],
) -> Result<Targets> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("cannot open targets file {}", path.display()))?;
    let headers = rdr.headers().context("cannot read CSV header")?.clone();
    let mut names: Vec<&str> = vec![x_col, y_col];
    names.extend(extra_cols.iter().map(String::as_str));
    let idx = column_indices(&headers, &names)?;

    let mut coords = Vec::new();
    let mut extras = Vec::new();
    let mut skipped = 0usize;
    for (row, record) in rdr.records().enumerate() {
        let record = record.with_context(|| format!("cannot parse CSV row {}", row + 2))?;
        let fields: Vec<Option<f64>> = idx.iter().map(|&i| parse_field(&record, i)).collect();
        if fields.iter().any(Option::is_none) {
            skipped += 1;
            continue;
        }
        let f: Vec<f64> = fields.into_iter().map(Option::unwrap).collect();
        coords.push([f[0], f[1]]);
        extras.push(f[2..].to_vec());
    }
    if skipped > 0 {
        eprintln!("warning: skipped {skipped} target rows with missing entries");
    }
    if coords.is_empty() {
        bail!("no valid target rows in {}", path.display());
    }
    Ok((coords, extras))
}

/// Writes experimental variogram bins as CSV.
pub fn write_variogram_csv(path: &Path, ev: &ExperimentalVariogram) -> Result<()> {
    let mut w = writer(path)?;
    writeln!(w, "lag,h,gamma,n_pairs")?;
    for (i, b) in ev.bins.iter().enumerate() {
        writeln!(w, "{},{},{},{}", i + 1, b.h, b.gamma, b.n_pairs)?;
    }
    Ok(())
}

/// Writes declustering weights as CSV (`x,y,value,weight`).
pub fn write_weights_csv(path: &Path, data: &PointSet, weights: &[f64]) -> Result<()> {
    let mut w = writer(path)?;
    writeln!(w, "x,y,value,weight")?;
    for (i, c) in data.coords().iter().enumerate() {
        writeln!(w, "{},{},{},{}", c[0], c[1], data.value(i), weights[i])?;
    }
    Ok(())
}

/// Writes gridded predictions as CSV (`x,y,prediction,variance`).
pub fn write_grid_csv(path: &Path, grid: &Grid2D, values: &[f64], variances: &[f64]) -> Result<()> {
    let mut w = writer(path)?;
    writeln!(w, "x,y,prediction,variance")?;
    for (i, c) in grid.centers().iter().enumerate() {
        writeln!(w, "{},{},{},{}", c[0], c[1], values[i], variances[i])?;
    }
    Ok(())
}

/// Writes per-point predictions as CSV (`x,y,prediction,variance`).
pub fn write_estimates_csv(
    path: &Path,
    coords: &[[f64; 2]],
    estimates: &[KrigingEstimate],
) -> Result<()> {
    let mut w = writer(path)?;
    writeln!(w, "x,y,prediction,variance")?;
    for (c, e) in coords.iter().zip(estimates) {
        writeln!(w, "{},{},{},{}", c[0], c[1], e.value, e.variance)?;
    }
    Ok(())
}

/// Writes simulation realizations as CSV (`x,y,sim1,...,simN`).
pub fn write_sims_csv(path: &Path, grid: &Grid2D, realizations: &[Vec<f64>]) -> Result<()> {
    let mut w = writer(path)?;
    let header: Vec<String> = (1..=realizations.len())
        .map(|i| format!("sim{i}"))
        .collect();
    writeln!(w, "x,y,{}", header.join(","))?;
    for (i, c) in grid.centers().iter().enumerate() {
        let row: Vec<String> = realizations.iter().map(|r| r[i].to_string()).collect();
        writeln!(w, "{},{},{}", c[0], c[1], row.join(","))?;
    }
    Ok(())
}

/// Writes leave-one-out residuals as CSV.
pub fn write_cv_csv(path: &Path, data: &PointSet, cv: &CvResult) -> Result<()> {
    let mut w = writer(path)?;
    writeln!(w, "x,y,observed,predicted,variance,residual")?;
    for i in 0..data.len() {
        let c = data.coord(i);
        writeln!(
            w,
            "{},{},{},{},{},{}",
            c[0],
            c[1],
            cv.observed[i],
            cv.predicted[i],
            cv.variance[i],
            cv.predicted[i] - cv.observed[i]
        )?;
    }
    Ok(())
}

/// Reads a variogram model from a JSON file.
pub fn read_model(path: &Path) -> Result<VariogramModel> {
    let file =
        File::open(path).with_context(|| format!("cannot open model file {}", path.display()))?;
    let model: VariogramModel = serde_json::from_reader(file)
        .with_context(|| format!("cannot parse model JSON in {}", path.display()))?;
    // Re-validate through the constructor.
    Ok(VariogramModel::new(model.nugget, model.structures)?)
}

/// Writes a variogram model to a JSON file.
pub fn write_model(path: &Path, model: &VariogramModel) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("cannot create model file {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), model)?;
    Ok(())
}

/// Reads a linear model of coregionalization from a JSON file.
pub fn read_lmc(path: &Path) -> Result<Lmc> {
    let file =
        File::open(path).with_context(|| format!("cannot open LMC file {}", path.display()))?;
    let lmc: Lmc = serde_json::from_reader(file)
        .with_context(|| format!("cannot parse LMC JSON in {}", path.display()))?;
    Ok(Lmc::new(lmc.nugget, lmc.structures)?)
}

/// Writes an LMC to a JSON file.
pub fn write_lmc(path: &Path, lmc: &Lmc) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("cannot create LMC file {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), lmc)?;
    Ok(())
}

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(File::create(path).with_context(|| {
        format!("cannot create output file {}", path.display())
    })?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // No `tempfile` dependency in this workspace: build unique paths under
    // the OS temp dir instead (pid + a per-process counter, so parallel
    // `#[test]` threads in this binary never collide).
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path(name: &str) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "geostat-io-utils-test-{}-{n}-{name}",
            std::process::id()
        ))
    }

    fn write_csv(path: &Path, contents: &str) {
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn read_points_happy_path() {
        let path = temp_path("points.csv");
        write_csv(&path, "x,y,z\n0,0,1.0\n1,0,2.0\n0,1,1.5\n");
        let data = read_points(&path, "x", "y", "z").unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data.coord(1), [1.0, 0.0]);
        assert_eq!(data.value(1), 2.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_points_reports_missing_column_and_lists_available() {
        let path = temp_path("points_bad_col.csv");
        write_csv(&path, "east,north,val\n0,0,1.0\n");
        let err = read_points(&path, "x", "y", "val").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("column 'x' not found"), "{msg}");
        assert!(msg.contains("east"), "should list available columns: {msg}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_points_skips_missing_or_non_numeric_rows() {
        let path = temp_path("points_missing.csv");
        write_csv(
            &path,
            "x,y,z\n0,0,1.0\n1,0,\n2,0,NaN\n3,0,na\n4,0,not_a_number\n5,0,2.0\n",
        );
        // 4 of 6 rows are unparseable/missing; only the two clean rows survive.
        let data = read_points(&path, "x", "y", "z").unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data.value(0), 1.0);
        assert_eq!(data.value(1), 2.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_points_errors_when_every_row_is_skipped() {
        let path = temp_path("points_all_missing.csv");
        write_csv(&path, "x,y,z\n0,0,\n1,0,na\n");
        let err = read_points(&path, "x", "y", "z").unwrap_err();
        assert!(format!("{err}").contains("no valid data rows"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_points_with_extras_reads_drift_columns() {
        let path = temp_path("points_extras.csv");
        write_csv(&path, "x,y,z,sdist\n0,0,1.0,0.1\n1,0,2.0,0.2\n");
        let (data, extras) =
            read_points_with_extras(&path, "x", "y", "z", &["sdist".to_string()]).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(extras, vec![vec![0.1], vec![0.2]]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_points3_assembles_xyz_and_value_correctly() {
        let path = temp_path("points3.csv");
        write_csv(&path, "x,y,z,grade\n0,0,5,1.0\n1,2,3,2.5\n");
        let data = read_points3(&path, "x", "y", "z", "grade").unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data.coord(0), [0.0, 0.0, 5.0]);
        assert_eq!(data.value(0), 1.0);
        assert_eq!(data.coord(1), [1.0, 2.0, 3.0]);
        assert_eq!(data.value(1), 2.5);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_targets_and_targets3_round_trip_coordinates_and_extras() {
        let path = temp_path("targets.csv");
        write_csv(&path, "x,y,sdist\n0,0,0.1\n1,1,0.2\n");
        let (coords, extras) = read_targets(&path, "x", "y", &["sdist".to_string()]).unwrap();
        assert_eq!(coords, vec![[0.0, 0.0], [1.0, 1.0]]);
        assert_eq!(extras, vec![vec![0.1], vec![0.2]]);
        std::fs::remove_file(&path).ok();

        let path3 = temp_path("targets3.csv");
        write_csv(&path3, "x,y,z\n0,0,5\n1,2,3\n");
        let targets3 = read_targets3(&path3, "x", "y", "z").unwrap();
        assert_eq!(targets3, vec![[0.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        std::fs::remove_file(&path3).ok();
    }

    #[test]
    fn model_json_round_trips_through_read_and_write() {
        let path = temp_path("model.json");
        let model = VariogramModel::new(
            0.1,
            vec![geostat_core::Structure::new(
                geostat_core::ModelKind::Spherical,
                0.9,
                10.0,
            )],
        )
        .unwrap();
        write_model(&path, &model).unwrap();
        let back = read_model(&path).unwrap();
        assert_eq!(back.nugget, model.nugget);
        assert_eq!(back.structures.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_model_reports_a_clear_error_for_a_missing_file() {
        let path = temp_path("does-not-exist.json");
        let err = read_model(&path).unwrap_err();
        assert!(format!("{err}").contains("cannot open model file"));
    }

    #[test]
    fn write_grid_csv_matches_grid_cell_count_and_order() {
        let path = temp_path("grid.csv");
        let grid = Grid2D::from_bbox([0.0, 0.0], [1.0, 1.0], 2, 2).unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let variances = vec![0.1, 0.2, 0.3, 0.4];
        write_grid_csv(&path, &grid, &values, &variances).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines[0], "x,y,prediction,variance");
        assert_eq!(lines.len(), 1 + grid.n_cells());
        std::fs::remove_file(&path).ok();
    }
}
