//! CSV input/output helpers for the CLI.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use geostat_core::{CvResult, ExperimentalVariogram, Grid2D, PointSet, VariogramModel};

/// Reads point data from a CSV file with named columns. Rows with empty,
/// "NA" or "NaN" entries in the requested columns are skipped with a warning.
pub fn read_points(path: &Path, x_col: &str, y_col: &str, value_col: &str) -> Result<PointSet> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("cannot open input file {}", path.display()))?;
    let headers = rdr.headers().context("cannot read CSV header")?.clone();

    let col = |name: &str| -> Result<usize> {
        headers.iter().position(|h| h == name).with_context(|| {
            format!(
                "column '{name}' not found; available columns: {}",
                headers.iter().collect::<Vec<_>>().join(", ")
            )
        })
    };
    let (xi, yi, vi) = (col(x_col)?, col(y_col)?, col(value_col)?);

    let mut coords = Vec::new();
    let mut values = Vec::new();
    let mut skipped = 0usize;
    for (row, record) in rdr.records().enumerate() {
        let record = record.with_context(|| format!("cannot parse CSV row {}", row + 2))?;
        let parse = |idx: usize| -> Option<f64> {
            let s = record.get(idx)?.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("na") || s.eq_ignore_ascii_case("nan") {
                return None;
            }
            s.parse().ok()
        };
        match (parse(xi), parse(yi), parse(vi)) {
            (Some(x), Some(y), Some(v)) => {
                coords.push([x, y]);
                values.push(v);
            }
            _ => skipped += 1,
        }
    }
    if skipped > 0 {
        eprintln!("warning: skipped {skipped} rows with missing or non-numeric entries");
    }
    if coords.is_empty() {
        bail!("no valid data rows in {}", path.display());
    }
    Ok(PointSet::new(coords, values)?)
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

/// Writes SGS realizations as CSV (`x,y,sim1,...,simN`).
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

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(File::create(path).with_context(|| {
        format!("cannot create output file {}", path.display())
    })?))
}
