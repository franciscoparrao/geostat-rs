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
