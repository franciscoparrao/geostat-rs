//! `geostat` — CLI for the geostat-rs geostatistics engine.

mod gpkg;
mod io_utils;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use geostat_core::{
    CoKriging, CoKrigingConfig, DirectionConfig, Grid2D, IkConfig, Kriging, KrigingConfig,
    KrigingMethod, ModelKind, PointSet, SgsConfig, SisConfig, Tails, VariogramConfig,
    VariogramModel, cell_declustering_weights, decluster_scan, detrend_external,
    detrend_polynomial, experimental_variogram, fit_anisotropic, fit_best, fit_indicator_models,
    fit_lmc_collocated, fit_median_indicator_model, indicator_kriging, k_fold, leave_one_out,
    leave_one_out_with_drift,
    sequential_gaussian_simulation, sequential_indicator_simulation, variogram_map, vecchia_mle,
    vecchia_predict, vecchia_reml,
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
    /// Compute a 2-D variogram map (lag-space surface) to reveal anisotropy
    Vmap(VmapCmd),
    /// Kriging interpolation (ordinary/simple/universal/external drift)
    Krige(KrigeCmd),
    /// Ordinary co-kriging with a secondary variable (LMC)
    Cokrige(CokrigeCmd),
    /// Leave-one-out cross-validation of a variogram model
    Cv(CvCmd),
    /// Conditional sequential Gaussian simulation
    Sgs(SgsCmd),
    /// Cell-declustering weights (GSLIB declus): scan cell sizes and export
    /// weights for preferentially sampled data
    Declus(DeclusCmd),
    /// Conditional sequential indicator simulation
    Sis(SisCmd),
    /// Indicator kriging: local ccdf, E-type estimate and conditional variance
    Ik(IkCmd),
    /// Regression kriging: OLS trend on covariates + kriging of residuals
    Rk(RkCmd),
    /// Compare interpolation methods by leave-one-out VEcv (OK, IDW, k-NN, NN)
    Compare(CompareCmd),
    /// Tune a method's hyperparameter by leave-one-out VEcv
    Tune(TuneCmd),
    /// List the vector feature and raster layers in a GeoPackage
    GpkgInfo(GpkgInfoCmd),
    /// Sample a GeoPackage raster (e.g. a DEM/NDVI covariate) at point locations
    GpkgSample(GpkgSampleCmd),
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
    /// GeoPackage layer name (when the input is a .gpkg with several layers)
    #[arg(long)]
    layer: Option<String>,
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
        if is_gpkg(&self.input) {
            return gpkg::read_points(&self.input, self.layer.as_deref(), &self.value_col);
        }
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
        VariogramConfig::for_data(
            data,
            self.n_lags,
            self.max_dist,
            self.azimuth.map(|az| DirectionConfig {
                azimuth_deg: az,
                dip_deg: self.dip,
                tolerance_deg: self.tolerance,
            }),
        )
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
    /// Minimum conditioning points per estimate (GSLIB ndmin): estimates
    /// with fewer neighbors fail instead of using too little data
    #[arg(long)]
    min_neighbors: Option<usize>,
    /// Maximum conditioning points per octant/quadrant around the target
    /// (GSLIB noct): balances the neighborhood for clustered data
    #[arg(long)]
    octant: Option<usize>,
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
    /// Fit a geometrically anisotropic model: estimate the major-axis azimuth
    /// and minor/major range ratio from directional variograms (2-D only).
    /// Combine with --fit to restrict the candidate families.
    #[arg(long)]
    anisotropic: bool,
    /// Number of directions for the anisotropy fit
    #[arg(long, default_value_t = 4)]
    n_dirs: usize,
    /// Fit the model by Vecchia maximum likelihood instead of weighted
    /// least squares (scalable; fits the covariance to the data likelihood).
    /// Uses the first --fit family, or exponential by default.
    #[arg(long)]
    mle: bool,
    /// Vecchia conditioning size for --mle
    #[arg(long, default_value_t = 20)]
    cond: usize,
    /// With --mle, fit by restricted/trend ML (REML) with a polynomial mean of
    /// this degree (0 = constant, 1 = linear, 2 = quadratic) instead of a
    /// constant plug-in mean. Use when the field has a spatial trend.
    #[arg(long)]
    trend: Option<u8>,
    /// Compute the variogram on OLS residuals of a polynomial trend in the
    /// coordinates (degree 1 or 2) — the correct variography for universal
    /// kriging (gstat's `z ~ x + y`)
    #[arg(long, value_name = "DEGREE")]
    detrend: Option<u8>,
    /// Compute the variogram on OLS residuals of a linear trend in these
    /// covariate columns (comma-separated) — the correct variography for
    /// kriging with an external drift
    #[arg(long, value_name = "COLS")]
    detrend_cols: Option<String>,
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
    /// Constant measurement-error variance added to the data-data diagonal
    /// (gstat Err): kriging predicts the signal and no longer honors the
    /// observations exactly
    #[arg(long)]
    error: Option<f64>,
    /// Column with a per-datum measurement-error variance (overrides --error)
    #[arg(long)]
    error_col: Option<String>,
    /// Vecchia prediction with this conditioning size (Katzfuss-Guinness):
    /// targets in max-min order condition on data and previous targets.
    /// Simple-kriging mean (data mean); scalable to very large n
    #[arg(long, value_name = "M")]
    vecchia: Option<usize>,
    #[command(flatten)]
    grid: GridOpts,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// Output file (x,y,prediction,variance). A .gpkg extension writes a
    /// GeoPackage point layer instead of CSV
    #[arg(short, long)]
    output: PathBuf,
    /// CRS (EPSG/srs_id) recorded when writing a GeoPackage output
    #[arg(long, default_value_t = 0)]
    srs: i32,
    /// For a grid .gpkg output, write a single-band raster (2D gridded
    /// coverage, prediction only) instead of a point layer
    #[arg(long)]
    raster: bool,
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
    /// Diagonal inflation of the co-kriging matrix (0 = exact system;
    /// e.g. 1e-2 stabilizes ill-conditioned heterotopic systems)
    #[arg(long, default_value_t = 0.0)]
    ridge: f64,
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
    /// Use k-fold cross-validation (k folds) instead of leave-one-out. Faster
    /// on large datasets; the split is reproducible via --seed. Not yet
    /// supported together with --drift-cols.
    #[arg(long, value_name = "K")]
    folds: Option<usize>,
    /// RNG seed for the k-fold split
    #[arg(long, default_value_t = 0)]
    seed: u64,
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
    /// Lower-tail extrapolation of the back-transform: none, linear or
    /// power:<w> (GSLIB ltail; requires --zmin)
    #[arg(long, default_value = "none")]
    ltail: String,
    /// Upper-tail extrapolation: none, linear, power:<w> or hyper:<w>
    /// (GSLIB utail; requires --zmax)
    #[arg(long, default_value = "none")]
    utail: String,
    /// Minimum attainable value for the lower tail (GSLIB zmin)
    #[arg(long)]
    zmin: Option<f64>,
    /// Maximum attainable value for the upper tail (GSLIB zmax)
    #[arg(long)]
    zmax: Option<f64>,
    /// Cell-declustering cell size: fit the normal-score reference
    /// distribution with declustering weights (see the `declus` subcommand
    /// to choose a size)
    #[arg(long)]
    declus: Option<f64>,
    /// Separate quota for previously simulated nodes (GSLIB nodmax): each
    /// neighborhood takes up to --max-neighbors data plus this many nodes,
    /// so simulated nodes cannot crowd out the hard data
    #[arg(long)]
    nodmax: Option<usize>,
    /// Multiple-grid simulation levels (GSLIB nmult): coarsest sub-grid
    /// (stride 2^levels) first, then refinements; improves long-range
    /// variogram reproduction on dense grids
    #[arg(long, default_value_t = 0)]
    multigrid: u8,
    /// Output CSV file (x,y,sim1..simN)
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct DeclusCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Single cell size (skips the scan)
    #[arg(long)]
    cell_size: Option<f64>,
    /// Smallest scanned cell size (default: bbox diagonal / 50)
    #[arg(long)]
    min_size: Option<f64>,
    /// Largest scanned cell size (default: bbox diagonal / 5)
    #[arg(long)]
    max_size: Option<f64>,
    /// Number of cell sizes to scan
    #[arg(long, default_value_t = 20)]
    n_sizes: usize,
    /// Grid-origin offsets averaged per size
    #[arg(long, default_value_t = 4)]
    offsets: usize,
    /// Keep the size that maximizes the declustered mean (default:
    /// minimize, for data preferentially clustered in high values)
    #[arg(long)]
    maximize: bool,
    /// Output CSV file (x,y,value,weight)
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
    /// Lower-tail interpolation: linear or power:<w> (GSLIB ltail)
    #[arg(long, default_value = "linear")]
    ltail: String,
    /// Upper-tail interpolation: linear, power:<w> or hyper:<w> (GSLIB
    /// utail; hyperbolic is capped at --tail-max)
    #[arg(long, default_value = "linear")]
    utail: String,
    /// Median IK (GSLIB mik=1): fit and krige with a single shared
    /// indicator variogram (the median cutoff's) instead of one per
    /// cutoff — amortizes one factorization across all cutoffs
    #[arg(long)]
    mik: bool,
    /// Ordinary indicator kriging (Σw=1) instead of simple IK (global
    /// proportion as the known mean)
    #[arg(long)]
    ordinary: bool,
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
    /// Lower-tail interpolation: linear or power:<w> (GSLIB ltail)
    #[arg(long, default_value = "linear")]
    ltail: String,
    /// Upper-tail interpolation: linear, power:<w> or hyper:<w> (GSLIB
    /// utail; hyperbolic is capped at --tail-max)
    #[arg(long, default_value = "linear")]
    utail: String,
    /// Median IK (GSLIB mik=1): fit and krige with a single shared
    /// indicator variogram (the median cutoff's) instead of one per
    /// cutoff — amortizes one factorization across all cutoffs. Ignored
    /// when --models is given explicitly.
    #[arg(long)]
    mik: bool,
    /// Ordinary indicator kriging (Σw=1) instead of simple IK (global
    /// proportion as the known mean)
    #[arg(long)]
    ordinary: bool,
    /// Output CSV file (x,y,F1..FK,e_type,cond_var)
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Variogram(cmd) => run_variogram(cmd),
        Command::Vmap(cmd) => run_vmap(cmd),
        Command::Krige(cmd) => run_krige(cmd),
        Command::Cokrige(cmd) => run_cokrige(cmd),
        Command::Cv(cmd) => run_cv(cmd),
        Command::Sgs(cmd) => run_sgs(cmd),
        Command::Declus(cmd) => run_declus(cmd),
        Command::Sis(cmd) => run_sis(cmd),
        Command::Ik(cmd) => run_ik(cmd),
        Command::Rk(cmd) => run_rk(cmd),
        Command::Compare(cmd) => run_compare(cmd),
        Command::Tune(cmd) => run_tune(cmd),
        Command::GpkgInfo(cmd) => run_gpkg_info(cmd),
        Command::GpkgSample(cmd) => run_gpkg_sample(cmd),
    }
}

/// True if the path looks like a GeoPackage (`.gpkg`, case-insensitive).
fn is_gpkg(path: &std::path::Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gpkg"))
}

/// Writes a kriged grid as CSV, or — when the path is a `.gpkg` — as a
/// GeoPackage point layer (cell centers, prediction + variance), or, with
/// `raster`, as a single-band 2D-gridded-coverage raster (prediction only).
fn write_grid_result(
    path: &std::path::Path,
    grid: &Grid2D,
    values: &[f64],
    variances: &[f64],
    srs: i32,
    raster: bool,
) -> Result<()> {
    if is_gpkg(path) {
        if raster {
            let bbox = [
                grid.x0,
                grid.y0,
                grid.x0 + grid.nx as f64 * grid.dx,
                grid.y0 + grid.ny as f64 * grid.dy,
            ];
            gpkg::write_raster(path, "kriging", srs, grid.nx, grid.ny, bbox, values)
        } else {
            gpkg::write_points(
                path,
                "kriging",
                srs,
                &grid.centers(),
                &[("prediction", values), ("variance", variances)],
            )
        }
    } else {
        io_utils::write_grid_csv(path, grid, values, variances)
    }
}

/// Writes kriging estimates at explicit targets as CSV, or — for a `.gpkg`
/// path — as a GeoPackage point layer with prediction and variance columns.
fn write_estimates_result(
    path: &std::path::Path,
    coords: &[[f64; 2]],
    ests: &[geostat_core::KrigingEstimate],
    srs: i32,
) -> Result<()> {
    if is_gpkg(path) {
        let values: Vec<f64> = ests.iter().map(|e| e.value).collect();
        let variances: Vec<f64> = ests.iter().map(|e| e.variance).collect();
        gpkg::write_points(
            path,
            "kriging",
            srs,
            coords,
            &[("prediction", &values), ("variance", &variances)],
        )
    } else {
        io_utils::write_estimates_csv(path, coords, ests)
    }
}

#[derive(Args)]
struct GpkgInfoCmd {
    /// GeoPackage file to inspect
    #[arg(short, long)]
    input: PathBuf,
}

#[derive(Args)]
struct GpkgSampleCmd {
    /// GeoPackage raster (2D gridded coverage) to sample
    #[arg(short, long)]
    raster: PathBuf,
    /// Raster layer name (when the file has several coverages)
    #[arg(long)]
    layer: Option<String>,
    /// CSV of point locations to sample at
    #[arg(short, long)]
    points: PathBuf,
    /// X coordinate column in the points CSV
    #[arg(long, default_value = "x")]
    x_col: String,
    /// Y coordinate column in the points CSV
    #[arg(long, default_value = "y")]
    y_col: String,
    /// Output CSV (x,y,<value>); points outside the extent or on no-data are
    /// written with an empty value
    #[arg(short, long)]
    output: PathBuf,
    /// Name of the sampled-value column in the output
    #[arg(long, default_value = "value")]
    value_col: String,
}

fn run_gpkg_sample(cmd: GpkgSampleCmd) -> Result<()> {
    let grid = gpkg::read_raster(&cmd.raster, cmd.layer.as_deref())?;
    let (coords, _) = io_utils::read_targets(&cmd.points, &cmd.x_col, &cmd.y_col, &[])?;

    let mut out = String::new();
    out.push_str(&format!("{},{},{}\n", cmd.x_col, cmd.y_col, cmd.value_col));
    let mut hit = 0usize;
    for c in &coords {
        match grid.sample(c[0], c[1]) {
            Some(v) => {
                hit += 1;
                out.push_str(&format!("{},{},{}\n", c[0], c[1], v));
            }
            None => out.push_str(&format!("{},{},\n", c[0], c[1])),
        }
    }
    std::fs::write(&cmd.output, out)
        .with_context(|| format!("writing {}", cmd.output.display()))?;
    println!(
        "Sampled raster '{}' ({}x{}) at {} points: {hit} hit, {} outside/no-data -> {}",
        grid.name,
        grid.nx,
        grid.ny,
        coords.len(),
        coords.len() - hit,
        cmd.output.display()
    );
    Ok(())
}

fn run_gpkg_info(cmd: GpkgInfoCmd) -> Result<()> {
    let layers = gpkg::list_feature_layers(&cmd.input)?;
    let rasters = gpkg::list_raster_layers(&cmd.input)?;
    if layers.is_empty() && rasters.is_empty() {
        println!("No feature or raster layers in {}", cmd.input.display());
        return Ok(());
    }
    if !layers.is_empty() {
        println!("Feature layers in {}:", cmd.input.display());
        println!(
            "  {:<24}{:<12}{:<14}{:>10}{:>10}",
            "layer", "geom_col", "type", "srs_id", "features"
        );
        for l in &layers {
            println!(
                "  {:<24}{:<12}{:<14}{:>10}{:>10}",
                l.name, l.geometry_column, l.geometry_type, l.srs_id, l.n_features
            );
        }
    }
    if !rasters.is_empty() {
        println!(
            "Raster (2D gridded coverage) layers in {}:",
            cmd.input.display()
        );
        println!("  {:<24}{:<14}{:>10}", "layer", "size", "srs_id");
        for name in &rasters {
            match gpkg::read_raster(&cmd.input, Some(name)) {
                Ok(g) => println!(
                    "  {:<24}{:<14}{:>10}",
                    g.name,
                    format!("{}x{}", g.nx, g.ny),
                    g.srs_id
                ),
                Err(e) => println!("  {name:<24}(unreadable: {e})"),
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, ValueEnum)]
enum TuneMethod {
    /// IDW power
    Idw,
    /// k-nearest-neighbor k
    Knn,
    /// Ordinary-kriging search-neighborhood size
    Ok,
}

#[derive(Args)]
struct TuneCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Which method's hyperparameter to tune
    #[arg(long, value_enum)]
    method: TuneMethod,
    /// Candidate values to try (comma-separated; floats for idw, ints for
    /// knn/ok). Defaults: idw 0.5,1,1.5,2,2.5,3,4,5; knn 1,2,3,4,6,8,12,16,24;
    /// ok 4,8,12,16,24,32,48
    #[arg(long)]
    grid: Option<String>,
    #[command(flatten)]
    vario: VariogramOpts,
    /// Search radius (optional)
    #[arg(long)]
    radius: Option<f64>,
}

fn run_tune(cmd: TuneCmd) -> Result<()> {
    use geostat_core::{tune_idw_power, tune_knn_k, tune_kriging_neighbors};

    let data = cmd.input.read()?;
    println!("Loaded {} points; tuning by leave-one-out VEcv", data.len());

    let ints = |default: &[usize]| -> Result<Vec<usize>> {
        match &cmd.grid {
            Some(s) => s
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse::<usize>()
                        .context("invalid integer in --grid")
                })
                .collect(),
            None => Ok(default.to_vec()),
        }
    };

    match cmd.method {
        TuneMethod::Idw => {
            let powers: Vec<f64> = match &cmd.grid {
                Some(s) => parse_floats(s)?,
                None => vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            };
            let res = tune_idw_power(&data, &powers, None, cmd.radius)?;
            print_tune_trace("IDW power", &res.trace, res.best_vecv);
            println!("Best IDW power = {} (VEcv {:.2}%)", res.best, res.best_vecv);
        }
        TuneMethod::Knn => {
            let ks = ints(&[1, 2, 3, 4, 6, 8, 12, 16, 24])?;
            let res = tune_knn_k(&data, &ks, cmd.radius)?;
            print_tune_trace("k-NN k", &res.trace, res.best_vecv);
            println!("Best k = {} (VEcv {:.2}%)", res.best, res.best_vecv);
        }
        TuneMethod::Ok => {
            let cfg = cmd.vario.config(&data);
            let ev = experimental_variogram(&data, &cfg)?;
            let model = fit_best(&ev, &ModelKind::ALL)?.model;
            println!("  fitted variogram: {model}");
            let cands = ints(&[4, 8, 12, 16, 24, 32, 48])?;
            let res =
                tune_kriging_neighbors(&data, &model, KrigingMethod::Ordinary, &cands, cmd.radius)?;
            print_tune_trace("OK neighbors", &res.trace, res.best_vecv);
            println!(
                "Best neighborhood size = {} (VEcv {:.2}%)",
                res.best, res.best_vecv
            );
        }
    }
    Ok(())
}

fn print_tune_trace<P: std::fmt::Display>(label: &str, trace: &[(P, f64)], best_vecv: f64) {
    println!("\n  {label:<14}{:>10}", "VEcv %");
    for (p, v) in trace {
        let mark = if (*v - best_vecv).abs() < 1e-12 {
            " <-"
        } else {
            ""
        };
        println!("  {p:<14}{v:>10.2}{mark}");
    }
    println!();
}

#[derive(Args)]
struct CompareCmd {
    #[command(flatten)]
    input: InputOpts,
    #[command(flatten)]
    vario: VariogramOpts,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// IDW power
    #[arg(long, default_value_t = 2.0)]
    idw_power: f64,
    /// k for k-nearest-neighbor averaging
    #[arg(long, default_value_t = 8)]
    knn_k: usize,
}

fn run_compare(cmd: CompareCmd) -> Result<()> {
    use geostat_core::{idw_cross_validate, knn_cross_validate};

    let data = cmd.input.read()?;
    println!(
        "Loaded {} points; comparing methods by leave-one-out",
        data.len()
    );

    // Ordinary kriging with an automatically fitted variogram.
    let cfg = cmd.vario.config(&data);
    let ev = experimental_variogram(&data, &cfg)?;
    let model = fit_best(&ev, &ModelKind::ALL)?.model;
    println!("  fitted variogram (for OK): {model}");
    let ok_config = KrigingConfig {
        method: KrigingMethod::Ordinary,
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
        min_neighbors: cmd.neighbors.min_neighbors,
        max_per_octant: cmd.neighbors.octant,
    };

    let mn = cmd.neighbors.max_neighbors;
    let rad = cmd.neighbors.radius;
    let mut results = vec![
        (
            "ordinary kriging".to_string(),
            leave_one_out(&data, &model, &ok_config)?,
        ),
        (
            format!("IDW (power {})", cmd.idw_power),
            idw_cross_validate(&data, cmd.idw_power, mn, rad)?,
        ),
        (
            format!("k-NN (k={})", cmd.knn_k),
            knn_cross_validate(&data, cmd.knn_k, rad)?,
        ),
        (
            "nearest neighbor".to_string(),
            knn_cross_validate(&data, 1, rad)?,
        ),
    ];
    // Rank by VEcv (higher is better).
    results.sort_by(|a, b| b.1.vecv().total_cmp(&a.1.vecv()));

    println!(
        "\n  {:<22}{:>8}{:>8}{:>10}{:>9}",
        "method (ranked)", "RMSE", "MAE", "VEcv %", "E1 %"
    );
    for (name, cv) in &results {
        println!(
            "  {name:<22}{:>8.4}{:>8.4}{:>10.2}{:>9.2}",
            cv.rmse(),
            cv.mae(),
            cv.vecv(),
            cv.e1()
        );
    }
    println!("\nVEcv = variance explained by cross-validation (Li 2016); higher is better.");
    Ok(())
}

#[derive(Args)]
struct RkCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Covariate columns for the OLS trend (comma-separated)
    #[arg(long)]
    covar_cols: String,
    /// Targets CSV with x, y and the same covariate columns
    #[arg(long)]
    targets: PathBuf,
    #[command(flatten)]
    vario: VariogramOpts,
    #[command(flatten)]
    neighbors: NeighborOpts,
    /// Output CSV file (x, y, prediction, variance)
    #[arg(short, long)]
    output: PathBuf,
}

fn run_rk(cmd: RkCmd) -> Result<()> {
    use geostat_core::{OlsTrend, RegressionKriging};

    let covar_cols: Vec<String> = cmd
        .covar_cols
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // 1. Fit the OLS trend on the data covariates.
    let (data, covars) = cmd.input.read_with_extras(&covar_cols)?;
    let trend = OlsTrend::fit(&covars, data.values())?;
    let trend_at_data: Vec<f64> = covars.iter().map(|c| trend.predict(c)).collect();
    println!(
        "Loaded {} points; OLS trend on [{}]",
        data.len(),
        covar_cols.join(", ")
    );
    let coefs = trend.coefficients();
    print!("  trend = {:.4}", coefs[0]);
    for (name, b) in covar_cols.iter().zip(&coefs[1..]) {
        print!(" + {b:.4}*{name}");
    }
    println!();

    // 2. Fit the residual variogram on z - m.
    let rk = RegressionKriging::new(&data, &trend_at_data)?;
    let cfg = cmd.vario.config(rk.residuals());
    let ev = experimental_variogram(rk.residuals(), &cfg)?;
    let resid_model = fit_best(&ev, &ModelKind::ALL)?.model;
    println!("  residual variogram: {resid_model}");

    // 3. Regression-krige the targets (trend + kriged residual).
    let (coords, target_covars) = io_utils::read_targets(
        &cmd.targets,
        &cmd.input.x_col,
        &cmd.input.y_col,
        &covar_cols,
    )?;
    let trend_at_targets: Vec<f64> = target_covars.iter().map(|c| trend.predict(c)).collect();
    let config = KrigingConfig {
        method: KrigingMethod::Ordinary,
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
        min_neighbors: cmd.neighbors.min_neighbors,
        max_per_octant: cmd.neighbors.octant,
    };
    let ests = rk.predict(&coords, &trend_at_targets, &resid_model, &config)?;
    let n_nan = ests.iter().filter(|e| e.value.is_nan()).count();
    println!(
        "Regression-kriged {} targets; {} failed",
        coords.len(),
        n_nan
    );
    io_utils::write_estimates_csv(&cmd.output, &coords, &ests)?;
    println!("Output written to {}", cmd.output.display());
    Ok(())
}
#[derive(Args)]
struct VmapCmd {
    #[command(flatten)]
    input: InputOpts,
    /// Number of lag cells on each side of the origin (map side = 2*n_lags+1)
    #[arg(long, default_value_t = 15)]
    n_lags: usize,
    /// Lag cell size in distance units. Default: a fifteenth of the data
    /// bounding-box half-diagonal.
    #[arg(long)]
    lag_width: Option<f64>,
    /// Output CSV (hx,hy,gamma,n_pairs)
    #[arg(short, long)]
    output: PathBuf,
}

fn run_vmap(cmd: VmapCmd) -> Result<()> {
    if cmd.input.z_col.is_some() {
        bail!("the variogram map is 2-D only (drop --z-col)");
    }
    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );
    let lag_width = match cmd.lag_width {
        Some(w) => w,
        None => {
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for c in data.coords() {
                for d in 0..2 {
                    lo[d] = lo[d].min(c[d]);
                    hi[d] = hi[d].max(c[d]);
                }
            }
            let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt();
            diag / 2.0 / cmd.n_lags as f64
        }
    };
    let m = variogram_map(&data, cmd.n_lags, lag_width)?;
    println!(
        "Variogram map {0}x{0}, lag width {1:.4}",
        m.size, m.lag_width
    );
    let mut out = String::from("hx,hy,gamma,n_pairs\n");
    for iy in 0..m.size {
        for ix in 0..m.size {
            let (hx, hy) = m.lag(ix, iy);
            let g = m.gamma_at(ix, iy);
            let np = m.n_pairs[iy * m.size + ix];
            let gstr = if g.is_finite() {
                g.to_string()
            } else {
                String::new()
            };
            out.push_str(&format!("{hx},{hy},{gstr},{np}\n"));
        }
    }
    std::fs::write(&cmd.output, out)
        .with_context(|| format!("writing {}", cmd.output.display()))?;
    println!("Variogram map written to {}", cmd.output.display());
    Ok(())
}

fn run_variogram(cmd: VariogramCmd) -> Result<()> {
    if cmd.anisotropic && cmd.mle {
        bail!("--mle and --anisotropic cannot be combined yet");
    }
    if cmd.trend.is_some() && !cmd.mle {
        bail!("--trend requires --mle (it selects REML/trend maximum likelihood)");
    }
    if cmd.detrend.is_some() && cmd.detrend_cols.is_some() {
        bail!("--detrend and --detrend-cols are mutually exclusive");
    }
    if cmd.input.z_col.is_some() {
        if cmd.anisotropic {
            bail!("--anisotropic is 2-D only (drop --z-col)");
        }
        if cmd.detrend_cols.is_some() {
            bail!("--detrend-cols is 2-D only (drop --z-col)");
        }
        let mut data = cmd.input.read3()?;
        println!("Loaded {} 3-D points", data.len());
        if let Some(deg) = cmd.detrend {
            data = detrend_polynomial(&data, deg)?.0;
            println!("Variogram computed on OLS residuals of a degree-{deg} trend");
        }
        return variogram_report(&data, &cmd);
    }
    let mut data = if let Some(cols) = &cmd.detrend_cols {
        let cols: Vec<String> = cols.split(',').map(|s| s.trim().to_string()).collect();
        let (raw, covars) = cmd.input.read_with_extras(&cols)?;
        let (resid, _) = detrend_external(&raw, &covars)?;
        println!(
            "Loaded {} points from {}; variogram on OLS residuals of drift [{}]",
            raw.len(),
            cmd.input.input.display(),
            cols.join(", ")
        );
        resid
    } else {
        let raw = cmd.input.read()?;
        println!(
            "Loaded {} points from {}",
            raw.len(),
            cmd.input.input.display()
        );
        raw
    };
    if let Some(deg) = cmd.detrend {
        data = detrend_polynomial(&data, deg)?.0;
        println!("Variogram computed on OLS residuals of a degree-{deg} trend");
    }
    if cmd.anisotropic {
        return anisotropic_report(&data, &cmd);
    }
    variogram_report(&data, &cmd)
}

fn anisotropic_report(data: &PointSet<2>, cmd: &VariogramCmd) -> Result<()> {
    let cfg = cmd.vario.config(data);
    let ev = experimental_variogram(data, &cfg)?;
    println!(
        "\nOmnidirectional variogram (max_dist = {:.2}):",
        cfg.max_dist
    );
    println!("{:>4} {:>12} {:>12} {:>8}", "lag", "h", "gamma", "pairs");
    for (i, b) in ev.bins.iter().enumerate() {
        let gamma = if b.n_pairs > 0 {
            format!("{:.6}", b.gamma)
        } else {
            "NA".to_string()
        };
        println!("{:>4} {:>12.2} {:>12} {:>8}", i + 1, b.h, gamma, b.n_pairs);
    }

    let kinds = match &cmd.fit {
        Some(spec) => parse_kinds(spec)?,
        None => ModelKind::ALL.to_vec(),
    };
    let fit = fit_anisotropic(data, &kinds, cmd.n_dirs, cfg.n_lags, cfg.max_dist)?;
    let s = fit.model.structures[0];
    let a = s
        .anis
        .expect("anisotropy fit always produces an anisotropic structure");
    println!("\nFitted anisotropic model ({} directions):", cmd.n_dirs);
    println!("  family:          {}", s.kind);
    println!("  nugget:          {:.4}", fit.model.nugget);
    println!("  partial sill:    {:.4}", s.sill);
    println!("  major range:     {:.4}", s.range);
    println!("  minor range:     {:.4}", s.range * a.ratio);
    println!(
        "  major azimuth:   {:.2} deg (clockwise from north)",
        a.azimuth_deg
    );
    println!("  ratio (min/maj): {:.4}", a.ratio);
    println!("  weighted SSE:    {:.6e}", fit.wsse);
    if a.ratio > 0.95 {
        println!("  note: ratio ~ 1 -> data look near-isotropic; azimuth is not meaningful");
    }
    if let Some(path) = &cmd.model_out {
        io_utils::write_model(path, &fit.model)?;
        println!("Model written to {}", path.display());
    }
    if let Some(path) = &cmd.output {
        io_utils::write_variogram_csv(path, &ev)?;
        println!("Omnidirectional bins written to {}", path.display());
    }
    Ok(())
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

    if cmd.mle {
        let kind = match &cmd.fit {
            Some(spec) => parse_kinds(spec)?[0],
            None => ModelKind::Exponential,
        };
        let (fit, label) = match cmd.trend {
            Some(deg) => (
                vecchia_reml(data, kind, cmd.cond, deg, None)?,
                format!("Vecchia REML fit (m = {}, trend degree {deg})", cmd.cond),
            ),
            None => (
                vecchia_mle(data, kind, cmd.cond, None)?,
                format!("Vecchia ML fit (m = {})", cmd.cond),
            ),
        };
        println!("\n{label}: {}", fit.model);
        println!("Log-likelihood: {:.4}", fit.loglik);
        if let Some(path) = &cmd.model_out {
            io_utils::write_model(path, &fit.model)?;
            println!("Model written to {}", path.display());
        }
    } else if let Some(spec) = &cmd.fit {
        let kinds = parse_kinds(spec)?;
        let fit = fit_best(&ev, &kinds)?;
        println!("\nFitted model: {}", fit.model);
        println!("Weighted SSE: {:.6e}", fit.wsse);
        if let Some(path) = &cmd.model_out {
            io_utils::write_model(path, &fit.model)?;
            println!("Model written to {}", path.display());
        }
    } else if cmd.model_out.is_some() {
        bail!("--model-out requires --fit or --mle");
    }

    if let Some(path) = &cmd.output {
        io_utils::write_variogram_csv(path, &ev)?;
        println!("Bins written to {}", path.display());
    }
    Ok(())
}

fn run_krige(cmd: KrigeCmd) -> Result<()> {
    let model = io_utils::read_model(&cmd.model)?;
    let has_error = cmd.error.is_some() || cmd.error_col.is_some();
    if has_error && (cmd.drift_cols.is_some() || cmd.lognormal || cmd.input.z_col.is_some()) {
        bail!("--error/--error-col support plain 2-D (block) kriging only for now");
    }
    if cmd.vecchia.is_some()
        && (has_error
            || cmd.drift_cols.is_some()
            || cmd.lognormal
            || cmd.block.is_some()
            || cmd.input.z_col.is_some())
    {
        bail!("--vecchia supports plain 2-D point kriging only (simple-kriging mean)");
    }

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
            min_neighbors: cmd.neighbors.min_neighbors,
            max_per_octant: cmd.neighbors.octant,
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
            min_neighbors: cmd.neighbors.min_neighbors,
            max_per_octant: cmd.neighbors.octant,
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
        write_estimates_result(&cmd.output, &coords, &ests, cmd.srs)?;
        println!("Output written to {}", cmd.output.display());
        return Ok(());
    }

    let (data, errors) = if let Some(col) = &cmd.error_col {
        let (data, extras) = cmd.input.read_with_extras(std::slice::from_ref(col))?;
        let errors: Vec<f64> = extras.into_iter().map(|row| row[0]).collect();
        (data, Some(errors))
    } else {
        let data = cmd.input.read()?;
        let errors = cmd.error.map(|e| vec![e; data.len()]);
        (data, errors)
    };
    println!("Loaded {} points; model: {model}", data.len());

    let grid = cmd.grid.build(&data)?;
    let config = KrigingConfig {
        method: cmd.method.build(&data),
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
        min_neighbors: cmd.neighbors.min_neighbors,
        max_per_octant: cmd.neighbors.octant,
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
        write_grid_result(&cmd.output, &grid, &values, &variances, cmd.srs, cmd.raster)?;
        println!("Output written to {}", cmd.output.display());
        return Ok(());
    }

    if let Some(m) = cmd.vecchia {
        let centers = grid.centers();
        let ests = vecchia_predict(&data, &model, &centers, m)?;
        let values: Vec<f64> = ests.iter().map(|e| e.value).collect();
        let variances: Vec<f64> = ests.iter().map(|e| e.variance).collect();
        println!(
            "Vecchia prediction on {} cells (m = {m}, simple-kriging mean)",
            grid.n_cells()
        );
        write_grid_result(&cmd.output, &grid, &values, &variances, cmd.srs, cmd.raster)?;
        println!("Output written to {}", cmd.output.display());
        return Ok(());
    }

    let kriging = match errors {
        Some(errors) => {
            println!("Measurement error active: kriging predicts the signal (not exact)");
            Kriging::with_measurement_error(&data, &model, config, errors)?
        }
        None => Kriging::new(&data, &model, config)?,
    };
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

    write_grid_result(&cmd.output, &grid, &values, &variances, cmd.srs, cmd.raster)?;
    println!("Output written to {}", cmd.output.display());
    Ok(())
}

fn run_cokrige(cmd: CokrigeCmd) -> Result<()> {
    if cmd.neighbors.min_neighbors.is_some() || cmd.neighbors.octant.is_some() {
        bail!("--min-neighbors/--octant are not supported by this command yet");
    }
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
            let lmc = fit_lmc_collocated(&primary, &secondary, &cfg, &ModelKind::ALL)?;
            println!("LMC auto-fitted:");
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
        ridge: cmd.ridge,
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

/// Prints the leave-one-out cross-validation report: error measures plus the
/// scale-free relative measures and the predictive-accuracy measures VEcv and
/// E₁ (Li 2016, 2017).
fn print_cv_report(cv: &geostat_core::CvResult, n: usize) {
    println!("\nLeave-one-out cross-validation ({n} points):");
    println!("  Mean error (bias): {:>12.6}", cv.mean_error());
    println!("  MAE:               {:>12.6}", cv.mae());
    println!("  RMSE:              {:>12.6}", cv.rmse());
    println!("  MSDR (ideal ~1):   {:>12.6}", cv.msdr());
    println!("  RME  (%):          {:>12.4}", cv.rme());
    println!("  RMAE (%):          {:>12.4}", cv.rmae());
    println!("  RRMSE (%):         {:>12.4}", cv.rrmse());
    println!("  VEcv (%, ideal 100): {:>10.4}", cv.vecv());
    println!("  E1   (%, ideal 100): {:>10.4}", cv.e1());
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
            min_neighbors: cmd.neighbors.min_neighbors,
            max_per_octant: cmd.neighbors.octant,
        };
        let cv = match cmd.folds {
            Some(k) => k_fold(&data, &model, &config, k, cmd.seed)?,
            None => leave_one_out(&data, &model, &config)?,
        };
        print_cv_report(&cv, data.len());
        if cmd.output.is_some() {
            bail!("--output is not supported in 3-D mode yet");
        }
        return Ok(());
    }

    let (data, cv) = if let Some(drift_spec) = &cmd.drift_cols {
        if cmd.folds.is_some() {
            bail!("--folds (k-fold) is not yet supported with --drift-cols");
        }
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
            min_neighbors: cmd.neighbors.min_neighbors,
            max_per_octant: cmd.neighbors.octant,
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
            min_neighbors: cmd.neighbors.min_neighbors,
            max_per_octant: cmd.neighbors.octant,
        };
        let cv = match cmd.folds {
            Some(k) => k_fold(&data, &model, &config, k, cmd.seed)?,
            None => leave_one_out(&data, &model, &config)?,
        };
        (data, cv)
    };

    print_cv_report(&cv, data.len());

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

    let decluster_weights = match cmd.declus {
        Some(size) => {
            let w = cell_declustering_weights(&data, size, 4)?;
            let mean = w
                .iter()
                .zip(data.values())
                .map(|(&wi, &v)| wi * v)
                .sum::<f64>()
                / data.len() as f64;
            println!(
                "Declustering (cell {size}): naive mean {:.4}, declustered mean {mean:.4}",
                data.mean()
            );
            Some(w)
        }
        None => None,
    };
    let grid = cmd.grid.build(&data)?;
    let cfg = SgsConfig {
        n_realizations: cmd.realizations,
        seed: cmd.seed,
        max_neighbors: cmd.max_neighbors,
        search_radius: cmd.radius,
        tails: Tails {
            lower: cmd.ltail.parse()?,
            upper: cmd.utail.parse()?,
            lower_bound: cmd.zmin,
            upper_bound: cmd.zmax,
        },
        decluster_weights,
        max_node_neighbors: cmd.nodmax,
        multigrid: cmd.multigrid,
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

fn run_declus(cmd: DeclusCmd) -> Result<()> {
    let data = cmd.input.read()?;
    println!(
        "Loaded {} points from {}",
        data.len(),
        cmd.input.input.display()
    );
    let naive = data.mean();
    let (weights, size, mean) = match cmd.cell_size {
        Some(size) => {
            let w = cell_declustering_weights(&data, size, cmd.offsets)?;
            let mean = w
                .iter()
                .zip(data.values())
                .map(|(&wi, &v)| wi * v)
                .sum::<f64>()
                / data.len() as f64;
            (w, size, mean)
        }
        None => {
            let (min, max) = data.bbox();
            let diag = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2)).sqrt();
            let lo = cmd.min_size.unwrap_or(diag / 50.0);
            let hi = cmd.max_size.unwrap_or(diag / 5.0);
            let scan = decluster_scan(&data, lo, hi, cmd.n_sizes, cmd.offsets, !cmd.maximize)?;
            println!(
                "Scanned {} sizes in [{lo:.3}, {hi:.3}] ({} origin offsets):",
                cmd.n_sizes, cmd.offsets
            );
            for (s, m) in &scan.trace {
                println!("  cell {s:>10.3} -> declustered mean {m:.5}");
            }
            (scan.weights, scan.best_size, scan.best_mean)
        }
    };
    println!("Naive mean:       {naive:.5}");
    println!("Declustered mean: {mean:.5} (cell size {size:.3})");
    io_utils::write_weights_csv(&cmd.output, &data, &weights)?;
    println!("Weights written to {}", cmd.output.display());
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

    // Fit an indicator variogram model per cutoff (or one shared model at
    // the median cutoff for --mik).
    let kinds = parse_kinds(&cmd.fit)?;
    let cfg_v = cmd.vario.config(&data);
    let models = if cmd.mik {
        fit_median_indicator_model(&data, &cutoffs, &kinds, &cfg_v)?
    } else {
        fit_indicator_models(&data, &cutoffs, &kinds, &cfg_v)?
    };
    if cmd.mik {
        println!("Median IK: shared model {}", models[0]);
    } else {
        for (c, m) in cutoffs.iter().zip(&models) {
            println!("Indicator model at cutoff {c:.4}: {m}");
        }
    }

    let grid = cmd.grid.build(&data)?;
    let cfg = SisConfig {
        cutoffs,
        models,
        ordinary: cmd.ordinary,
        n_realizations: cmd.realizations,
        seed: cmd.seed,
        max_neighbors: cmd.max_neighbors,
        search_radius: cmd.radius,
        tail_min: cmd.tail_min,
        tail_max: cmd.tail_max,
        lower_tail: cmd.ltail.parse()?,
        upper_tail: cmd.utail.parse()?,
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
    if cmd.neighbors.min_neighbors.is_some() || cmd.neighbors.octant.is_some() {
        bail!("--min-neighbors/--octant are not supported by this command yet");
    }
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
            if cmd.mik {
                let models = fit_median_indicator_model(&data, &cutoffs, &kinds, &cfg_v)?;
                println!("Median IK: shared model {}", models[0]);
                models
            } else {
                let models = fit_indicator_models(&data, &cutoffs, &kinds, &cfg_v)?;
                for (c, m) in cutoffs.iter().zip(&models) {
                    println!("Indicator model at cutoff {c:.4}: {m}");
                }
                models
            }
        }
    };

    let grid = cmd.grid.build(&data)?;
    let n_cutoffs = cutoffs.len();
    let cfg = IkConfig {
        cutoffs,
        models,
        ordinary: cmd.ordinary,
        max_neighbors: cmd.neighbors.max_neighbors,
        search_radius: cmd.neighbors.radius,
        tail_min: cmd.tail_min,
        tail_max: cmd.tail_max,
        lower_tail: cmd.ltail.parse()?,
        upper_tail: cmd.utail.parse()?,
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
    ModelKind::parse_list(spec).map_err(Into::into)
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
