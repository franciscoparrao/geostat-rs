//! `geostat` — CLI for the geostat-rs geostatistics engine.

mod io_utils;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use geostat_core::{
    DirectionConfig, Grid2D, Kriging, KrigingConfig, KrigingMethod, ModelKind, PointSet, SgsConfig,
    VariogramConfig, experimental_variogram, fit_best, leave_one_out,
    sequential_gaussian_simulation,
};

#[derive(Parser)]
#[command(
    name = "geostat",
    version,
    about = "Geostatistics engine: variography, kriging and sequential Gaussian simulation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute an experimental variogram and optionally fit a model
    Variogram(VariogramCmd),
    /// Kriging interpolation onto a regular grid
    Krige(KrigeCmd),
    /// Leave-one-out cross-validation of a variogram model
    Cv(CvCmd),
    /// Conditional sequential Gaussian simulation
    Sgs(SgsCmd),
}

#[derive(Args)]
struct InputOpts {
    /// Input CSV file with point data
    #[arg(short, long)]
    input: PathBuf,
    /// Column name for the X coordinate
    #[arg(long, default_value = "x")]
    x_col: String,
    /// Column name for the Y coordinate
    #[arg(long, default_value = "y")]
    y_col: String,
    /// Column name for the variable of interest
    #[arg(long, default_value = "z")]
    value_col: String,
}

impl InputOpts {
    fn read(&self) -> Result<PointSet> {
        io_utils::read_points(&self.input, &self.x_col, &self.y_col, &self.value_col)
    }
}

#[derive(Args)]
struct VariogramOpts {
    /// Number of lag bins
    #[arg(long, default_value_t = 15)]
    n_lags: usize,
    /// Maximum pair distance (default: bbox diagonal / 3)
    #[arg(long)]
    max_dist: Option<f64>,
    /// Azimuth in degrees clockwise from north, for a directional variogram
    #[arg(long)]
    azimuth: Option<f64>,
    /// Angular tolerance in degrees for the directional variogram
    #[arg(long, default_value_t = 22.5)]
    tolerance: f64,
}

impl VariogramOpts {
    fn config(&self, data: &PointSet) -> VariogramConfig {
        let (min, max) = data.bbox();
        let diag = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2)).sqrt();
        VariogramConfig {
            n_lags: self.n_lags,
            max_dist: self.max_dist.unwrap_or(diag / 3.0),
            direction: self.azimuth.map(|az| DirectionConfig {
                azimuth_deg: az,
                tolerance_deg: self.tolerance,
            }),
        }
    }
}

#[derive(Args)]
struct GridOpts {
    /// Bounding box "xmin,ymin,xmax,ymax" (default: data bbox)
    #[arg(long)]
    bbox: Option<String>,
    /// Number of grid columns
    #[arg(long, default_value_t = 100)]
    nx: usize,
    /// Number of grid rows
    #[arg(long, default_value_t = 100)]
    ny: usize,
    /// Square cell size (overrides nx/ny)
    #[arg(long)]
    res: Option<f64>,
}

impl GridOpts {
    fn build(&self, data: &PointSet) -> Result<Grid2D> {
        let (min, max) = match &self.bbox {
            Some(s) => parse_bbox(s)?,
            None => data.bbox(),
        };
        let grid = match self.res {
            Some(res) => Grid2D::with_resolution(min, max, res)?,
            None => Grid2D::from_bbox(min, max, self.nx, self.ny)?,
        };
        Ok(grid)
    }
}

#[derive(Args)]
struct NeighborOpts {
    /// Maximum number of nearest neighbors per estimate (default: all points)
    #[arg(long)]
    max_neighbors: Option<usize>,
    /// Search radius for conditioning points
    #[arg(long)]
    radius: Option<f64>,
}

#[derive(Clone, Copy, ValueEnum)]
enum MethodArg {
    Ordinary,
    Simple,
    Universal,
}

#[derive(Args)]
struct MethodOpts {
    /// Kriging method
    #[arg(long, value_enum, default_value_t = MethodArg::Ordinary)]
    method: MethodArg,
    /// Drift polynomial degree for universal kriging (1 or 2)
    #[arg(long, default_value_t = 1)]
    degree: u8,
    /// Known mean for simple kriging (default: data mean)
    #[arg(long)]
    mean: Option<f64>,
}

impl MethodOpts {
    fn build(&self, data: &PointSet) -> KrigingMethod {
        match self.method {
            MethodArg::Ordinary => KrigingMethod::Ordinary,
            MethodArg::Simple => KrigingMethod::Simple {
                mean: self.mean.unwrap_or_else(|| data.mean()),
            },
            MethodArg::Universal => KrigingMethod::Universal {
                degree: self.degree,
            },
        }
    }
}

#[derive(Args)]
struct VariogramCmd {
    #[command(flatten)]
    input: InputOpts,
    #[command(flatten)]
    vario: VariogramOpts,
    /// Fit model(s): "best", or comma-separated kinds
    /// (spherical, exponential, gaussian, matern15, matern25)
    #[arg(long)]
    fit: Option<String>,
    /// Write experimental variogram bins to a CSV file
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Write the fitted model to a JSON file
    #[arg(long)]
    model_out: Option<PathBuf>,
}

#[derive(Args)]
struct KrigeCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Fitted variogram model (JSON, as written by `variogram --model-out`)
    #[arg(short, long)]
    model: PathBuf,
    #[command(flatten)]
    method: MethodOpts,
    #[command(flatten)]
    grid: GridOpts,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// Output CSV file (x,y,prediction,variance)
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct CvCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Fitted variogram model (JSON)
    #[arg(short, long)]
    model: PathBuf,
    #[command(flatten)]
    method: MethodOpts,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// Write per-point residuals to a CSV file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct SgsCmd {
    #[command(flatten)]
    input: InputOpts,
    #[command(flatten)]
    grid: GridOpts,
    #[command(flatten)]
    vario: VariogramOpts,
    /// Variogram model for the normal scores (JSON). If omitted, a model is
    /// fitted automatically to the normal-score variogram
    #[arg(long)]
    model_ns: Option<PathBuf>,
    /// Model kinds to try when auto-fitting ("best" or comma-separated list)
    #[arg(long, default_value = "best")]
    fit: String,
    /// Number of realizations
    #[arg(short = 'n', long, default_value_t = 10)]
    realizations: usize,
    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Maximum conditioning points per node
    #[arg(long, default_value_t = 16)]
    max_neighbors: usize,
    /// Search radius for conditioning points
    #[arg(long)]
    radius: Option<f64>,
    /// Output CSV file (x,y,sim1..simN)
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Variogram(cmd) => run_variogram(cmd),
        Command::Krige(cmd) => run_krige(cmd),
        Command::Cv(cmd) => run_cv(cmd),
        Command::Sgs(cmd) => run_sgs(cmd),
    }
}

fn run_variogram(cmd: VariogramCmd) -> Result<()> {
    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );

    let cfg = cmd.vario.config(&data);
    let ev = experimental_variogram(&data, &cfg)?;

    println!("\nExperimental variogram (max_dist = {:.2}):", cfg.max_dist);
    println!("{:>4} {:>12} {:>12} {:>8}", "lag", "h", "gamma", "pairs");
    for (i, b) in ev.bins.iter().enumerate() {
        let gamma = if b.n_pairs > 0 {
            format!("{:.6}", b.gamma)
        } else {
            "NA".to_string()
        };
        println!("{:>4} {:>12.2} {:>12} {:>8}", i + 1, b.h, gamma, b.n_pairs);
    }

    if let Some(spec) = &cmd.fit {
        let kinds = parse_kinds(spec)?;
        let fit = fit_best(&ev, &kinds)?;
        println!("\nFitted model: {}", fit.model);
        println!("Weighted SSE: {:.6e}", fit.wsse);
        if let Some(path) = &cmd.model_out {
            io_utils::write_model(path, &fit.model)?;
            println!("Model written to {}", path.display());
        }
    } else if cmd.model_out.is_some() {
        bail!("--model-out requires --fit");
    }

    if let Some(path) = &cmd.output {
        io_utils::write_variogram_csv(path, &ev)?;
        println!("Bins written to {}", path.display());
    }
    Ok(())
}

fn run_krige(cmd: KrigeCmd) -> Result<()> {
    let data = cmd.input.read()?;
    let model = io_utils::read_model(&cmd.model)?;
    println!("Loaded {} points; model: {}", data.len(), model);

    let grid = cmd.grid.build(&data)?;
    let config = KrigingConfig {
        method: cmd.method.build(&data),
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
    };
    let kriging = Kriging::new(&data, &model, config)?;
    let (values, variances) = kriging.predict_grid(&grid);

    let n_nan = values.iter().filter(|v| v.is_nan()).count();
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    println!(
        "Kriged {} cells ({} x {}); {} empty (no neighbors)",
        grid.n_cells(),
        grid.nx,
        grid.ny,
        n_nan
    );
    if !finite.is_empty() {
        let min = finite.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = finite.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        println!("Prediction range: [{min:.4}, {max:.4}]");
    }

    io_utils::write_grid_csv(&cmd.output, &grid, &values, &variances)?;
    println!("Output written to {}", cmd.output.display());
    Ok(())
}

fn run_cv(cmd: CvCmd) -> Result<()> {
    let data = cmd.input.read()?;
    let model = io_utils::read_model(&cmd.model)?;
    println!("Loaded {} points; model: {}", data.len(), model);

    let config = KrigingConfig {
        method: cmd.method.build(&data),
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
    };
    let cv = leave_one_out(&data, &model, &config)?;

    println!("\nLeave-one-out cross-validation ({} points):", data.len());
    println!("  Mean error (bias): {:>12.6}", cv.mean_error());
    println!("  MAE:               {:>12.6}", cv.mae());
    println!("  RMSE:              {:>12.6}", cv.rmse());
    println!("  MSDR (ideal ~1):   {:>12.6}", cv.msdr());

    if let Some(path) = &cmd.output {
        io_utils::write_cv_csv(path, &data, &cv)?;
        println!("Residuals written to {}", path.display());
    }
    Ok(())
}

fn run_sgs(cmd: SgsCmd) -> Result<()> {
    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );

    let model_ns = match &cmd.model_ns {
        Some(path) => {
            let m = io_utils::read_model(path)?;
            println!("Normal-score model (from file): {m}");
            m
        }
        None => {
            // Fit a model to the variogram of the normal scores.
            let ns = geostat_core::NormalScore::fit(data.values())?;
            let scores: Vec<f64> = data.values().iter().map(|&v| ns.transform(v)).collect();
            let ns_data = PointSet::new(data.coords().to_vec(), scores)?;
            let cfg = cmd.vario.config(&ns_data);
            let ev = experimental_variogram(&ns_data, &cfg)?;
            let kinds = parse_kinds(&cmd.fit)?;
            let fit = fit_best(&ev, &kinds)?;
            println!("Normal-score model (auto-fitted): {}", fit.model);
            fit.model
        }
    };

    let grid = cmd.grid.build(&data)?;
    let cfg = SgsConfig {
        n_realizations: cmd.realizations,
        seed: cmd.seed,
        max_neighbors: cmd.max_neighbors,
        search_radius: cmd.radius,
    };
    let res = sequential_gaussian_simulation(&data, &model_ns, &grid, &cfg)?;

    println!(
        "Simulated {} realizations on {} cells ({} x {}), seed {}",
        cfg.n_realizations,
        grid.n_cells(),
        grid.nx,
        grid.ny,
        cfg.seed
    );
    io_utils::write_sims_csv(&cmd.output, &grid, &res.realizations)?;
    println!("Output written to {}", cmd.output.display());
    Ok(())
}

fn parse_kinds(spec: &str) -> Result<Vec<ModelKind>> {
    let spec = spec.trim();
    if spec.eq_ignore_ascii_case("best") || spec.eq_ignore_ascii_case("all") {
        return Ok(ModelKind::ALL.to_vec());
    }
    spec.split(',')
        .map(|s| s.parse::<ModelKind>().map_err(Into::into))
        .collect()
}

fn parse_bbox(s: &str) -> Result<([f64; 2], [f64; 2])> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|p| p.trim().parse::<f64>().context("invalid bbox number"))
        .collect::<Result<_>>()?;
    if parts.len() != 4 {
        bail!("bbox must be \"xmin,ymin,xmax,ymax\", got '{s}'");
    }
    Ok(([parts[0], parts[1]], [parts[2], parts[3]]))
}
