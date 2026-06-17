//! `geostat` — CLI for the geostat-rs geostatistics engine.

mod io_utils;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use geostat_core::{
    CoKriging, CoKrigingConfig, DirectionConfig, Grid2D, IkConfig, Kriging, KrigingConfig,
    KrigingMethod, ModelKind, PointSet, SgsConfig, SisConfig, VariogramConfig, VariogramModel,
    experimental_cross_variogram, experimental_variogram, fit_best, fit_lmc, indicator_kriging,
    leave_one_out, leave_one_out_with_drift, sequential_gaussian_simulation,
    sequential_indicator_simulation,
};

#[derive(Parser)]
#[command(
    name = "geostat",
    version,
    about = "Geostatistics engine: variography, kriging, co-kriging and sequential simulation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute an experimental variogram and optionally fit a model
    Variogram(VariogramCmd),
    /// Kriging interpolation (ordinary/simple/universal/external drift)
    Krige(KrigeCmd),
    /// Ordinary co-kriging with a secondary variable (LMC)
    Cokrige(CokrigeCmd),
    /// Leave-one-out cross-validation of a variogram model
    Cv(CvCmd),
    /// Conditional sequential Gaussian simulation
    Sgs(SgsCmd),
    /// Conditional sequential indicator simulation
    Sis(SisCmd),
    /// Indicator kriging: local ccdf, E-type estimate and conditional variance
    Ik(IkCmd),
    /// Transport (warped) kriging: learnable marginal warp + latent kriging
    Tgp(TgpCmd),
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
    /// Column name for the Z coordinate: switches 3-D mode on
    /// (variogram, krige with --targets, cv)
    #[arg(long)]
    z_col: Option<String>,
    /// Column name for the variable of interest
    #[arg(long, default_value = "z")]
    value_col: String,
}

impl InputOpts {
    fn read3(&self) -> Result<PointSet<3>> {
        let z_col = self.z_col.as_ref().expect("z_col checked by caller");
        io_utils::read_points3(
            &self.input,
            &self.x_col,
            &self.y_col,
            z_col,
            &self.value_col,
        )
    }

    fn read(&self) -> Result<PointSet> {
        io_utils::read_points(&self.input, &self.x_col, &self.y_col, &self.value_col)
    }

    fn read_with_extras(&self, extras: &[String]) -> Result<(PointSet, Vec<Vec<f64>>)> {
        io_utils::read_points_with_extras(
            &self.input,
            &self.x_col,
            &self.y_col,
            &self.value_col,
            extras,
        )
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
    /// Dip in degrees (positive downward) for 3-D directional variograms
    #[arg(long, default_value_t = 0.0)]
    dip: f64,
}

impl VariogramOpts {
    fn config<const D: usize>(&self, data: &PointSet<D>) -> VariogramConfig {
        let (min, max) = data.bbox();
        let diag = (0..D)
            .map(|d| (max[d] - min[d]).powi(2))
            .sum::<f64>()
            .sqrt();
        VariogramConfig {
            n_lags: self.n_lags,
            max_dist: self.max_dist.unwrap_or(diag / 3.0),
            direction: self.azimuth.map(|az| DirectionConfig {
                azimuth_deg: az,
                dip_deg: self.dip,
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
    /// Kriging method (ignored when --drift-cols is given: external drift)
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
    fn build<const D: usize>(&self, data: &PointSet<D>) -> KrigingMethod {
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
    /// External drift columns (comma-separated; requires --targets)
    #[arg(long)]
    drift_cols: Option<String>,
    /// Targets CSV with x, y and the drift columns (external drift only)
    #[arg(long)]
    targets: Option<PathBuf>,
    /// Block kriging: block size "width,height" centered on each cell
    #[arg(long)]
    block: Option<String>,
    /// Block discretization points per axis "nx,ny"
    #[arg(long, default_value = "4,4")]
    block_discr: String,
    /// Lognormal kriging: data values are positive, model is of ln(value);
    /// predictions are back-transformed to original units
    #[arg(long)]
    lognormal: bool,
    #[command(flatten)]
    grid: GridOpts,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// Output CSV file (x,y,prediction,variance)
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct CokrigeCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Column with the secondary variable
    #[arg(long)]
    secondary_col: String,
    /// Separate CSV with the secondary variable (heterotopic co-kriging;
    /// same coordinate column names). Requires --lmc
    #[arg(long)]
    secondary_input: Option<PathBuf>,
    /// LMC model (JSON). If omitted, an LMC is fitted automatically from the
    /// direct and cross variograms (collocated data only)
    #[arg(long)]
    lmc: Option<PathBuf>,
    /// Write the (fitted) LMC to a JSON file
    #[arg(long)]
    lmc_out: Option<PathBuf>,
    #[command(flatten)]
    vario: VariogramOpts,
    /// Block co-kriging: block size "width,height" centered on each cell
    #[arg(long)]
    block: Option<String>,
    /// Block discretization points per axis "nx,ny"
    #[arg(long, default_value = "4,4")]
    block_discr: String,
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
    /// External drift columns (comma-separated): cross-validate KED
    #[arg(long)]
    drift_cols: Option<String>,
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

#[derive(Args)]
struct SisCmd {
    #[command(flatten)]
    input: InputOpts,
    #[command(flatten)]
    grid: GridOpts,
    #[command(flatten)]
    vario: VariogramOpts,
    /// Indicator cutoffs as data quantiles, comma-separated
    #[arg(long, default_value = "0.25,0.5,0.75")]
    quantiles: String,
    /// Explicit indicator cutoffs (overrides --quantiles)
    #[arg(long)]
    cutoffs: Option<String>,
    /// Model kinds to try when fitting indicator variograms
    #[arg(long, default_value = "spherical,exponential")]
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
    /// Lower tail bound (default: data minimum)
    #[arg(long)]
    tail_min: Option<f64>,
    /// Upper tail bound (default: data maximum)
    #[arg(long)]
    tail_max: Option<f64>,
    /// Output CSV file (x,y,sim1..simN)
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct IkCmd {
    #[command(flatten)]
    input: InputOpts,
    #[command(flatten)]
    grid: GridOpts,
    #[command(flatten)]
    vario: VariogramOpts,
    /// Indicator cutoffs as data quantiles, comma-separated
    #[arg(long, default_value = "0.25,0.5,0.75")]
    quantiles: String,
    /// Explicit indicator cutoffs (overrides --quantiles)
    #[arg(long)]
    cutoffs: Option<String>,
    /// Indicator model JSONs, comma-separated, one per cutoff
    /// (default: auto-fit per cutoff)
    #[arg(long)]
    models: Option<String>,
    /// Model kinds to try when auto-fitting indicator variograms
    #[arg(long, default_value = "spherical,exponential")]
    fit: String,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// Lower tail bound (default: data minimum)
    #[arg(long)]
    tail_min: Option<f64>,
    /// Upper tail bound (default: data maximum)
    #[arg(long)]
    tail_max: Option<f64>,
    /// Output CSV file (x,y,F1..FK,e_type,cond_var)
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Clone, Copy, ValueEnum)]
enum WarpArg {
    /// Pick the family automatically by AIC (default)
    Auto,
    /// No warp (plain ordinary kriging with Monte Carlo quantiles)
    Identity,
    /// Box–Cox power transform (auto-fitted; positive or shiftable data)
    BoxCox,
    /// Yeo–Johnson power transform (auto-fitted; any-sign data)
    YeoJohnson,
    /// sinh–arcsinh transform (auto-fitted skew + tails)
    SinhArcsinh,
}

#[derive(Args)]
struct TgpCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Marginal transport family to fit
    #[arg(long, value_enum, default_value_t = WarpArg::Auto)]
    warp: WarpArg,
    /// Clamp back-transformed predictions to be at least this value (e.g.
    /// `--floor 0` for a non-negative quantity under a real-line warp)
    #[arg(long)]
    floor: Option<f64>,
    #[command(flatten)]
    vario: VariogramOpts,
    #[command(flatten)]
    grid: GridOpts,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// Monte Carlo samples per cell for the back-transform
    #[arg(long, default_value_t = 2000)]
    samples: usize,
    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Output CSV file (x,y,mean,std)
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Variogram(cmd) => run_variogram(cmd),
        Command::Krige(cmd) => run_krige(cmd),
        Command::Cokrige(cmd) => run_cokrige(cmd),
        Command::Cv(cmd) => run_cv(cmd),
        Command::Sgs(cmd) => run_sgs(cmd),
        Command::Sis(cmd) => run_sis(cmd),
        Command::Ik(cmd) => run_ik(cmd),
        Command::Tgp(cmd) => run_tgp(cmd),
    }
}

fn run_tgp(cmd: TgpCmd) -> Result<()> {
    use geostat_core::{
        FittedMarginal, Identity, MarginalTransport, TransportKriging, fit_best_marginal,
        fit_box_cox, fit_sinh_arcsinh, fit_yeo_johnson,
    };

    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );

    // Generic driver: fit the chosen marginal, fit the latent variogram on
    // the warped data, then warped-krige the grid.
    fn run<T: MarginalTransport + Sync>(
        data: &PointSet,
        marginal: FittedMarginal<T>,
        cmd: &TgpCmd,
    ) -> Result<()> {
        let marginal = match cmd.floor {
            Some(f) => marginal.with_floor(f),
            None => marginal,
        };
        let latent_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&z| marginal.to_latent(z))
            .collect();
        let latent = PointSet::new(data.coords().to_vec(), latent_vals)?;
        let cfg = cmd.vario.config(&latent);
        let ev = experimental_variogram(&latent, &cfg)?;
        let fit = fit_best(&ev, &ModelKind::ALL)?;
        println!("Latent variogram: {}", fit.model);

        let grid = cmd.grid.build(data)?;
        let config = KrigingConfig {
            method: KrigingMethod::Ordinary,
            max_neighbors: cmd.neighbors.max_neighbors,
            search_radius: cmd.neighbors.radius,
        };
        let tk = TransportKriging::new(data, marginal, &fit.model, config)?;
        let (means, stds) = tk.predict_grid(&grid, cmd.samples, cmd.seed)?;
        println!(
            "Warped kriging on {} cells ({} x {}), {} MC samples/cell",
            grid.n_cells(),
            grid.nx,
            grid.ny,
            cmd.samples
        );
        io_utils::write_grid_csv(&cmd.output, &grid, &means, &stds)?;
        println!("Output written to {}", cmd.output.display());
        Ok(())
    }

    match cmd.warp {
        WarpArg::Auto => {
            let sel = fit_best_marginal(data.values())?;
            println!("AIC selection (lower is better):");
            for (name, aic) in &sel.candidates {
                let mark = if *name == sel.family {
                    " <- selected"
                } else {
                    ""
                };
                println!("  {name:<13} AIC = {aic:10.3}{mark}");
            }
            run(&data, sel.marginal, &cmd)
        }
        WarpArg::Identity => {
            println!("No warp (identity): plain ordinary kriging with MC quantiles");
            run(&data, FittedMarginal::new(Identity, 0.0, 1.0)?, &cmd)
        }
        WarpArg::BoxCox => {
            let m = fit_box_cox(data.values())?;
            println!("Fitted Box–Cox: lambda = {:.4}", m.transform().lambda);
            run(&data, m, &cmd)
        }
        WarpArg::YeoJohnson => {
            let m = fit_yeo_johnson(data.values())?;
            println!("Fitted Yeo–Johnson: lambda = {:.4}", m.transform().lambda);
            run(&data, m, &cmd)
        }
        WarpArg::SinhArcsinh => {
            let m = fit_sinh_arcsinh(data.values())?;
            println!(
                "Fitted sinh–arcsinh: epsilon = {:.4}, delta = {:.4}",
                m.transform().epsilon,
                m.transform().delta
            );
            run(&data, m, &cmd)
        }
    }
}

fn run_variogram(cmd: VariogramCmd) -> Result<()> {
    if cmd.input.z_col.is_some() {
        let data = cmd.input.read3()?;
        println!("Loaded {} 3-D points", data.len());
        return variogram_report(&data, &cmd);
    }
    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );
    variogram_report(&data, &cmd)
}

fn variogram_report<const D: usize>(data: &PointSet<D>, cmd: &VariogramCmd) -> Result<()> {
    let cfg = cmd.vario.config(data);
    let ev = experimental_variogram(data, &cfg)?;

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
    let model = io_utils::read_model(&cmd.model)?;

    if let Some(z_col) = &cmd.input.z_col {
        // 3-D kriging at explicit targets.
        if cmd.drift_cols.is_some() || cmd.block.is_some() {
            bail!("3-D mode supports plain point kriging only (no --drift-cols/--block)");
        }
        let targets_path = cmd
            .targets
            .as_ref()
            .context("3-D kriging requires --targets (CSV with x, y, z)")?;
        let data = cmd.input.read3()?;
        println!("Loaded {} 3-D points; model: {model}", data.len());
        let config = KrigingConfig {
            method: cmd.method.build(&data),
            max_neighbors: cmd.neighbors.max_neighbors,
            search_radius: cmd.neighbors.radius,
        };
        let kriging: Kriging<'_, 3> = Kriging::new(&data, &model, config)?;
        let targets =
            io_utils::read_targets3(targets_path, &cmd.input.x_col, &cmd.input.y_col, z_col)?;
        let ests = kriging.predict_many(&targets);
        let n_nan = ests.iter().filter(|e| e.value.is_nan()).count();
        println!("Kriged {} 3-D targets; {} failed", targets.len(), n_nan);
        io_utils::write_estimates3_csv(&cmd.output, &targets, &ests)?;
        println!("Output written to {}", cmd.output.display());
        return Ok(());
    }

    if let Some(drift_spec) = &cmd.drift_cols {
        // External drift kriging at explicit targets.
        let drift_cols: Vec<String> = drift_spec
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let targets_path = cmd
            .targets
            .as_ref()
            .context("--drift-cols requires --targets (CSV with x, y and drift columns)")?;
        let (data, drift_data) = cmd.input.read_with_extras(&drift_cols)?;
        println!(
            "Loaded {} points; model: {model}; drift: {}",
            data.len(),
            drift_cols.join(", ")
        );
        let config = KrigingConfig {
            method: KrigingMethod::ExternalDrift {
                n_vars: drift_cols.len(),
            },
            max_neighbors: cmd.neighbors.max_neighbors,
            search_radius: cmd.neighbors.radius,
        };
        let kriging = Kriging::with_external_drift(&data, &model, config, drift_data)?;
        let (coords, target_drift) = io_utils::read_targets(
            targets_path,
            &cmd.input.x_col,
            &cmd.input.y_col,
            &drift_cols,
        )?;
        let ests = kriging.predict_many_with_drift(&coords, &target_drift)?;
        let n_nan = ests.iter().filter(|e| e.value.is_nan()).count();
        println!(
            "Kriged {} target points (external drift); {} failed",
            coords.len(),
            n_nan
        );
        io_utils::write_estimates_csv(&cmd.output, &coords, &ests)?;
        println!("Output written to {}", cmd.output.display());
        return Ok(());
    }

    let data = cmd.input.read()?;
    println!("Loaded {} points; model: {model}", data.len());

    let grid = cmd.grid.build(&data)?;
    let config = KrigingConfig {
        method: cmd.method.build(&data),
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
    };

    if cmd.lognormal {
        if cmd.block.is_some() || cmd.drift_cols.is_some() {
            bail!("--lognormal cannot be combined with --block or --drift-cols");
        }
        let centers = grid.centers();
        let ests = geostat_core::lognormal_kriging(&data, &centers, &model, &config)?;
        let values: Vec<f64> = ests.iter().map(|e| e.value).collect();
        let variances: Vec<f64> = ests.iter().map(|e| e.log_variance).collect();
        println!(
            "Lognormal kriging on {} cells (variance column is in log space)",
            grid.n_cells()
        );
        io_utils::write_grid_csv(&cmd.output, &grid, &values, &variances)?;
        println!("Output written to {}", cmd.output.display());
        return Ok(());
    }

    let kriging = Kriging::new(&data, &model, config)?;
    let (values, variances) = match &cmd.block {
        Some(spec) => {
            let size = parse_floats(spec)?;
            let discr = parse_floats(&cmd.block_discr)?;
            if size.len() != 2 || discr.len() != 2 {
                bail!("--block and --block-discr take two comma-separated values");
            }
            println!(
                "Block kriging: {} x {} blocks, {} x {} discretization",
                size[0], size[1], discr[0], discr[1]
            );
            kriging.predict_block_grid(
                &grid,
                [size[0], size[1]],
                [discr[0] as usize, discr[1] as usize],
            )?
        }
        None => kriging.predict_grid(&grid),
    };

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

fn run_cokrige(cmd: CokrigeCmd) -> Result<()> {
    let (primary, secondary) = match &cmd.secondary_input {
        Some(path) => {
            // Heterotopic: the secondary variable has its own locations.
            if cmd.lmc.is_none() {
                bail!(
                    "--secondary-input (heterotopic co-kriging) requires --lmc: \
                     automatic LMC fitting needs collocated data"
                );
            }
            let primary = cmd.input.read()?;
            let secondary = io_utils::read_points(
                path,
                &cmd.input.x_col,
                &cmd.input.y_col,
                &cmd.secondary_col,
            )?;
            println!(
                "Loaded {} primary + {} secondary points (heterotopic)",
                primary.len(),
                secondary.len()
            );
            (primary, secondary)
        }
        None => {
            let (primary, extras) = cmd
                .input
                .read_with_extras(std::slice::from_ref(&cmd.secondary_col))?;
            let secondary = PointSet::new(
                primary.coords().to_vec(),
                extras.iter().map(|r| r[0]).collect(),
            )?;
            println!(
                "Loaded {} collocated points ({} + {})",
                primary.len(),
                cmd.input.value_col,
                cmd.secondary_col
            );
            (primary, secondary)
        }
    };

    let lmc = match &cmd.lmc {
        Some(path) => {
            let lmc = io_utils::read_lmc(path)?;
            println!("LMC loaded from {}", path.display());
            lmc
        }
        None => {
            let cfg = cmd.vario.config(&primary);
            let ea = experimental_variogram(&primary, &cfg)?;
            let eb = experimental_variogram(&secondary, &cfg)?;
            let eab = experimental_cross_variogram(&primary, &secondary, &cfg)?;
            let template = fit_best(&ea, &ModelKind::ALL)?;
            let lmc = fit_lmc(&ea, &eb, &eab, &template.model)?;
            println!(
                "LMC auto-fitted (template {} {}, range {:.1}):",
                template.model.structures[0].sill,
                template.model.structures[0].kind,
                template.model.structures[0].range,
            );
            println!("  nugget matrix: {:?}", lmc.nugget);
            for s in &lmc.structures {
                println!("  {} ({:.1}): {:?}", s.kind, s.range, s.sills);
            }
            lmc
        }
    };
    if let Some(path) = &cmd.lmc_out {
        io_utils::write_lmc(path, &lmc)?;
        println!("LMC written to {}", path.display());
    }

    let grid = cmd.grid.build(&primary)?;
    let config = CoKrigingConfig {
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
    };
    let ck = CoKriging::new(vec![&primary, &secondary], &lmc, config)?;
    let (values, variances) = match &cmd.block {
        Some(spec) => {
            let size = parse_floats(spec)?;
            let discr = parse_floats(&cmd.block_discr)?;
            if size.len() != 2 || discr.len() != 2 {
                bail!("--block and --block-discr take two comma-separated values");
            }
            println!(
                "Block co-kriging: {} x {} blocks, {} x {} discretization",
                size[0], size[1], discr[0], discr[1]
            );
            ck.predict_block_grid(
                &grid,
                [size[0], size[1]],
                [discr[0] as usize, discr[1] as usize],
            )?
        }
        None => ck.predict_grid(&grid),
    };
    let n_nan = values.iter().filter(|v| v.is_nan()).count();
    println!(
        "Co-kriged {} cells ({} x {}); {} empty",
        grid.n_cells(),
        grid.nx,
        grid.ny,
        n_nan
    );
    io_utils::write_grid_csv(&cmd.output, &grid, &values, &variances)?;
    println!("Output written to {}", cmd.output.display());
    Ok(())
}

fn run_cv(cmd: CvCmd) -> Result<()> {
    let model = io_utils::read_model(&cmd.model)?;

    if cmd.input.z_col.is_some() {
        if cmd.drift_cols.is_some() {
            bail!("3-D cross-validation does not support --drift-cols");
        }
        let data = cmd.input.read3()?;
        println!("Loaded {} 3-D points; model: {model}", data.len());
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
        if cmd.output.is_some() {
            bail!("--output is not supported in 3-D mode yet");
        }
        return Ok(());
    }

    let (data, cv) = if let Some(drift_spec) = &cmd.drift_cols {
        let drift_cols: Vec<String> = drift_spec
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let (data, drift_data) = cmd.input.read_with_extras(&drift_cols)?;
        println!(
            "Loaded {} points; model: {model}; drift: {}",
            data.len(),
            drift_cols.join(", ")
        );
        let config = KrigingConfig {
            method: KrigingMethod::ExternalDrift {
                n_vars: drift_cols.len(),
            },
            max_neighbors: cmd.neighbors.max_neighbors,
            search_radius: cmd.neighbors.radius,
        };
        let cv = leave_one_out_with_drift(&data, &drift_data, &model, &config)?;
        (data, cv)
    } else {
        let data = cmd.input.read()?;
        println!("Loaded {} points; model: {model}", data.len());
        let config = KrigingConfig {
            method: cmd.method.build(&data),
            max_neighbors: cmd.neighbors.max_neighbors,
            search_radius: cmd.neighbors.radius,
        };
        let cv = leave_one_out(&data, &model, &config)?;
        (data, cv)
    };

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

fn run_sis(cmd: SisCmd) -> Result<()> {
    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );

    // Cutoffs: explicit values or data quantiles.
    let cutoffs: Vec<f64> = match &cmd.cutoffs {
        Some(spec) => parse_floats(spec)?,
        None => {
            let qs = parse_floats(&cmd.quantiles)?;
            let mut sorted = data.values().to_vec();
            sorted.sort_by(f64::total_cmp);
            qs.iter()
                .map(|&q| {
                    if !(0.0..=1.0).contains(&q) {
                        bail!("quantile {q} outside [0, 1]");
                    }
                    let idx = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
                    Ok(sorted[idx])
                })
                .collect::<Result<_>>()?
        }
    };
    println!(
        "Cutoffs: {}",
        cutoffs
            .iter()
            .map(|c| format!("{c:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Fit an indicator variogram model per cutoff.
    let kinds = parse_kinds(&cmd.fit)?;
    let cfg_v = cmd.vario.config(&data);
    let mut models: Vec<VariogramModel> = Vec::with_capacity(cutoffs.len());
    for &c in &cutoffs {
        let indicators: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v <= c { 1.0 } else { 0.0 })
            .collect();
        let ind_data = PointSet::new(data.coords().to_vec(), indicators)?;
        let ev = experimental_variogram(&ind_data, &cfg_v)?;
        let fit = fit_best(&ev, &kinds)?;
        println!("Indicator model at cutoff {c:.4}: {}", fit.model);
        models.push(fit.model);
    }

    let grid = cmd.grid.build(&data)?;
    let cfg = SisConfig {
        cutoffs,
        models,
        n_realizations: cmd.realizations,
        seed: cmd.seed,
        max_neighbors: cmd.max_neighbors,
        search_radius: cmd.radius,
        tail_min: cmd.tail_min,
        tail_max: cmd.tail_max,
    };
    let res = sequential_indicator_simulation(&data, &grid, &cfg)?;

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

fn run_ik(cmd: IkCmd) -> Result<()> {
    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );

    let cutoffs: Vec<f64> = match &cmd.cutoffs {
        Some(spec) => parse_floats(spec)?,
        None => {
            let qs = parse_floats(&cmd.quantiles)?;
            let mut sorted = data.values().to_vec();
            sorted.sort_by(f64::total_cmp);
            qs.iter()
                .map(|&q| {
                    if !(0.0..=1.0).contains(&q) {
                        bail!("quantile {q} outside [0, 1]");
                    }
                    let idx = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
                    Ok(sorted[idx])
                })
                .collect::<Result<_>>()?
        }
    };
    println!(
        "Cutoffs: {}",
        cutoffs
            .iter()
            .map(|c| format!("{c:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let models: Vec<VariogramModel> = match &cmd.models {
        Some(spec) => {
            let paths: Vec<&str> = spec.split(',').map(str::trim).collect();
            if paths.len() != cutoffs.len() {
                bail!("{} models for {} cutoffs", paths.len(), cutoffs.len());
            }
            paths
                .iter()
                .map(|p| io_utils::read_model(std::path::Path::new(p)))
                .collect::<Result<_>>()?
        }
        None => {
            let kinds = parse_kinds(&cmd.fit)?;
            let cfg_v = cmd.vario.config(&data);
            cutoffs
                .iter()
                .map(|&c| {
                    let indicators: Vec<f64> = data
                        .values()
                        .iter()
                        .map(|&v| if v <= c { 1.0 } else { 0.0 })
                        .collect();
                    let ind = PointSet::new(data.coords().to_vec(), indicators)?;
                    let ev = experimental_variogram(&ind, &cfg_v)?;
                    let fit = fit_best(&ev, &kinds)?;
                    println!("Indicator model at cutoff {c:.4}: {}", fit.model);
                    Ok(fit.model)
                })
                .collect::<Result<_>>()?
        }
    };

    let grid = cmd.grid.build(&data)?;
    let n_cutoffs = cutoffs.len();
    let cfg = IkConfig {
        cutoffs,
        models,
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
        tail_min: cmd.tail_min,
        tail_max: cmd.tail_max,
    };
    let centers = grid.centers();
    let ests = indicator_kriging(&data, &centers, &cfg)?;
    println!(
        "Indicator kriging on {} cells ({} x {}), {} cutoffs",
        grid.n_cells(),
        grid.nx,
        grid.ny,
        n_cutoffs
    );
    io_utils::write_ik_csv(&cmd.output, &centers, &ests, n_cutoffs)?;
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

fn parse_floats(spec: &str) -> Result<Vec<f64>> {
    spec.split(',')
        .map(|p| {
            p.trim()
                .parse::<f64>()
                .with_context(|| format!("invalid number '{}'", p.trim()))
        })
        .collect()
}

fn parse_bbox(s: &str) -> Result<([f64; 2], [f64; 2])> {
    let parts = parse_floats(s)?;
    if parts.len() != 4 {
        bail!("bbox must be \"xmin,ymin,xmax,ymax\", got '{s}'");
    }
    Ok(([parts[0], parts[1]], [parts[2], parts[3]]))
}
