"""Type stubs for geostat_rs — Python bindings for the geostat-rs
geostatistics engine (variography, kriging, co-kriging, sequential
simulation).

Array-shaped results (grid predictions, simulation realizations,
cross-validation arrays) are returned as numpy arrays.
"""

from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

__version__: str

class VariogramModel:
    """A fitted variogram model (JSON-compatible with the geostat CLI)."""

    @staticmethod
    def from_json(json: str) -> "VariogramModel":
        """Parses a model from its JSON representation."""

    def to_json(self) -> str:
        """JSON representation (usable with the geostat CLI)."""

    def gamma(self, h: float) -> float:
        """Semivariance at scalar lag ``h``."""

    def total_sill(self) -> float:
        """Total sill (nugget + partial sills)."""

    def anisotropy(self) -> Optional[tuple[float, float]]:
        """Geometric anisotropy of the first anisotropic structure as
        ``(major_azimuth_deg, minor_over_major_ratio)``, or ``None`` if
        isotropic."""

    def __repr__(self) -> str: ...

def experimental_variogram(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    azimuth: Optional[float] = None,
    tolerance: float = 22.5,
    detrend: Optional[int] = None,
    detrend_drift: Optional[Sequence[Sequence[float]]] = None,
    estimator: str = "matheron",
) -> tuple[list[float], list[float], list[int]]:
    """Experimental semivariogram. Returns ``(h, gamma, n_pairs)`` lists;
    empty bins carry NaN gamma. ``estimator``: "matheron" (default),
    "cressie-hawkins"/"ch", "dowd" or "madogram" -- the latter three are
    more resistant to a few outlier pairs."""

def variogram_map(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    lag_width: Optional[float] = None,
) -> dict[str, Any]:
    """2-D variogram map (lag-space semivariance surface). Returns a dict
    with ``size``, ``lag_width``, and flat row-major lists ``hx``, ``hy``,
    ``gamma``, ``n_pairs``. ``lag_width`` defaults to a fifteenth of the
    data's bounding-box half-diagonal when omitted."""

def fit_variogram(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    kinds: str = "best",
    detrend: Optional[int] = None,
    detrend_drift: Optional[Sequence[Sequence[float]]] = None,
) -> VariogramModel:
    """Fits a variogram model by weighted least squares."""

def fit_anisotropic(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    n_dirs: int = 4,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    kinds: str = "best",
) -> VariogramModel:
    """Fits a geometrically anisotropic variogram model."""

def vecchia_mle(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    kind: str = "exponential",
    m: int = 20,
) -> VariogramModel:
    """Fits a single-structure model by Vecchia maximum likelihood."""

def vecchia_reml(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    kind: str = "exponential",
    m: int = 20,
    drift_degree: int = 1,
) -> VariogramModel:
    """Fits a single-structure model by Vecchia restricted/trend maximum
    likelihood."""

def vecchia_reml_drift(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    covariates: Sequence[Sequence[float]],
    kind: str = "exponential",
    m: int = 20,
) -> VariogramModel:
    """Fits a single-structure model by Vecchia external-drift REML."""

def vecchia_param_se(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    m: int = 20,
) -> tuple[float, float, float]:
    """Asymptotic standard errors ``(se_nugget, se_sill, se_range)``."""

def vecchia_krige(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    target_x: Sequence[float],
    target_y: Sequence[float],
    m: int = 30,
) -> tuple[FloatArray, FloatArray]:
    """Vecchia prediction (Katzfuss-Guinness) at arbitrary targets. Returns
    ``(predictions, variances)``."""

def vecchia_loglik(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    m: int = 20,
) -> float:
    """Vecchia-approximated Gaussian log-likelihood of the data under
    ``model``."""

def krige(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    target_x: Sequence[float],
    target_y: Sequence[float],
    method: str = "ordinary",
    mean: Optional[float] = None,
    degree: int = 1,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    min_neighbors: Optional[int] = None,
    octant: Optional[int] = None,
    measurement_error: Optional[Sequence[float]] = None,
    search_azimuth: Optional[float] = None,
    search_ratio: float = 1.0,
    search_ratio_z: float = 1.0,
    search_dip: float = 0.0,
    search_rake: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Kriging at arbitrary target locations. Returns
    ``(predictions, variances)``; failed targets yield NaN. With
    ``search_azimuth`` set, searches a rotated ellipsoid (``radius`` becomes
    the major-axis radius) instead of a Euclidean neighborhood."""

def lognormal_kriging(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    log_model: VariogramModel,
    target_x: Sequence[float],
    target_y: Sequence[float],
    method: str = "ordinary",
    mean: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
) -> tuple[FloatArray, FloatArray]:
    """Ordinary/simple lognormal kriging. Returns
    ``(predictions, log_variances)``."""

def krige_grid(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    method: str = "ordinary",
    mean: Optional[float] = None,
    degree: int = 1,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    block: Optional[tuple[float, float]] = None,
    block_discr: tuple[int, int] = (4, 4),
    min_neighbors: Optional[int] = None,
    octant: Optional[int] = None,
    search_azimuth: Optional[float] = None,
    search_ratio: float = 1.0,
    search_ratio_z: float = 1.0,
    search_dip: float = 0.0,
    search_rake: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Kriging over a regular grid. Returns ``(predictions, variances)``,
    row-major with y increasing. With ``search_azimuth`` set, searches a
    rotated ellipsoid instead of a Euclidean neighborhood -- see ``krige``."""

def loo_cv(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    method: str = "ordinary",
    mean: Optional[float] = None,
    degree: int = 1,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    folds: Optional[int] = None,
    seed: int = 0,
    min_neighbors: Optional[int] = None,
    octant: Optional[int] = None,
    blocks: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    """Cross-validation (leave-one-out by default; ``folds``/``blocks`` for
    k-fold or spatial block CV). Returns a dict with ``me``, ``mae``,
    ``rmse``, ``msdr``, ``vecv``, ``e1``, ``observed``, ``predicted`` and
    ``variance``."""

def accuracy_plot(
    actual: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float],
    probs: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Deutsch (1997) accuracy plot. Returns a dict with ``nominal``,
    ``observed`` and ``goodness`` (1.0 = perfect calibration)."""

def regression_kriging(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    covariates: Sequence[Sequence[float]],
    target_x: Sequence[float],
    target_y: Sequence[float],
    target_covariates: Sequence[Sequence[float]],
    trend_at_data: Optional[Sequence[float]] = None,
    trend_at_targets: Optional[Sequence[float]] = None,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
) -> dict[str, Any]:
    """Regression kriging: OLS trend + kriged residuals. Returns a dict with
    ``prediction``, ``variance``, and (built-in trend only) ``trend_coef``."""

def co_kriging(
    px: Sequence[float],
    py: Sequence[float],
    pv: Sequence[float],
    sx: Sequence[float],
    sy: Sequence[float],
    sv: Sequence[float],
    target_x: Sequence[float],
    target_y: Sequence[float],
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    ridge: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Ordinary co-kriging of a primary variable using a correlated
    secondary (linear model of coregionalization, fitted automatically).
    Returns ``(predictions, variances)`` of the primary."""

def collocated_stats(
    primary: Sequence[float],
    secondary: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates ``(rho12, sigma1, sigma2)`` from collocated sample pairs
    (Pearson correlation and sample standard deviations) -- the usual
    inputs to ``collocated_cokriging`` when no external population
    estimate of these statistics is available."""

def collocated_cokriging(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    target_x: Sequence[float],
    target_y: Sequence[float],
    target_secondary: Sequence[float],
    mean1: float,
    mean2: float,
    rho12: float,
    sigma1: float,
    sigma2: float,
    markov: str = "mm1",
    secondary_model: Optional[VariogramModel] = None,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    ridge: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Collocated cokriging (MM1/MM2, Journel 1999): predicts the primary
    from its own moving neighborhood plus a single secondary value
    collocated with each target, via a Markov screening hypothesis instead
    of a fitted cross-variogram -- the practical choice for an
    exhaustively sampled secondary (raster/seismic). ``markov`` is
    ``"mm1"`` (default, needs only ``model``) or ``"mm2"`` (needs
    ``secondary_model``). Simple-kriging form only (known means: ``mean1``/
    ``mean2``). Returns ``(predictions, variances)`` of the primary at the
    targets; a target that fails yields NaN rather than aborting the batch."""

def idw(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    target_x: Sequence[float],
    target_y: Sequence[float],
    power: float = 2.0,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
) -> FloatArray:
    """Inverse-distance weighting at the targets."""

def knn(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    target_x: Sequence[float],
    target_y: Sequence[float],
    k: int = 8,
    radius: Optional[float] = None,
) -> FloatArray:
    """k-nearest-neighbor averaging at the targets (``k = 1`` is
    nearest-neighbor)."""

def compare_methods(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    idw_power: float = 2.0,
    knn_k: int = 8,
) -> dict[str, dict[str, float]]:
    """Compares interpolation methods by leave-one-out cross-validation.
    Returns ``method -> {rmse, mae, vecv, e1}``."""

def tune_idw_power(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    powers: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
) -> dict[str, Any]:
    """Tunes the IDW ``power`` by leave-one-out VEcv."""

def tune_knn_k(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    ks: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24),
    radius: Optional[float] = None,
) -> dict[str, Any]:
    """Tunes the k-NN ``k`` by leave-one-out VEcv."""

def tune_kriging_neighbors(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    candidates: Sequence[int] = (4, 8, 12, 16, 24, 32, 48),
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    radius: Optional[float] = None,
) -> dict[str, Any]:
    """Tunes the ordinary-kriging search-neighborhood size by leave-one-out
    VEcv."""

def decluster_weights(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    cell_size: Optional[float] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    n_sizes: int = 20,
    n_offsets: int = 4,
    minimize: bool = True,
) -> dict[str, Any]:
    """Cell declustering weights. Scans a range of cell sizes when
    ``cell_size`` is omitted. Returns a dict with ``weights``,
    ``cell_size``, ``declustered_mean``, ``trace``."""

def sgs(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    model_ns: VariogramModel,
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    n_realizations: int = 10,
    seed: int = 42,
    max_neighbors: int = 16,
    radius: Optional[float] = None,
    ltail: str = "none",
    utail: str = "none",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    decluster_cell: Optional[float] = None,
    max_node_neighbors: Optional[int] = None,
    multigrid: int = 0,
) -> FloatArray:
    """Conditional sequential Gaussian simulation. Returns an
    ``n_realizations x n_cells`` array, one row per realization."""

def tgs(
    x: Sequence[float],
    y: Sequence[float],
    categories: Sequence[int],
    model: VariogramModel,
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    n_categories: Optional[int] = None,
    proportions: Optional[Sequence[float]] = None,
    n_realizations: int = 10,
    seed: int = 42,
    max_neighbors: int = 16,
    radius: Optional[float] = None,
    decluster_cell: Optional[float] = None,
    max_node_neighbors: Optional[int] = None,
    multigrid: int = 0,
) -> FloatArray:
    """Truncated Gaussian simulation (TGS) for ordered categorical/facies
    data: one underlying Gaussian field truncated at thresholds derived
    from global category proportions. ``model`` must already be a
    variogram of the underlying standard-Gaussian field (sill ~ 1) -- not
    auto-fitted. Returns an ``n_realizations x n_cells`` array of category
    ids (as floats), one row per realization."""

def sis(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    cutoffs: Sequence[float],
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    n_realizations: int = 10,
    seed: int = 42,
    max_neighbors: int = 16,
    radius: Optional[float] = None,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    fit: str = "spherical,exponential",
    ltail: str = "linear",
    utail: str = "linear",
    tail_min: Optional[float] = None,
    tail_max: Optional[float] = None,
    mik: bool = False,
    ordinary: bool = False,
    decluster_cell: Optional[float] = None,
    max_node_neighbors: Optional[int] = None,
    multigrid: int = 0,
) -> FloatArray:
    """Conditional sequential indicator simulation. Returns an
    ``n_realizations x n_cells`` array, one row per realization. ``fit`` is
    a comma-separated list of candidate variogram families for the
    per-cutoff auto-fit (e.g. "spherical,exponential,matern:1.5").
    ``decluster_cell``/``max_node_neighbors``/``multigrid`` mirror ``sgs``'s
    same-named parameters."""

def experimental_variogram_3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    azimuth: Optional[float] = None,
    dip: float = 0.0,
    tolerance: float = 22.5,
) -> tuple[list[float], list[float], list[int]]:
    """3-D experimental semivariogram. Returns ``(h, gamma, n_pairs)``
    lists."""

def fit_variogram_3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    kinds: str = "best",
) -> VariogramModel:
    """Fits a variogram model to 3-D data by weighted least squares."""

def krige_3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    values: Sequence[float],
    model: VariogramModel,
    target_x: Sequence[float],
    target_y: Sequence[float],
    target_z: Sequence[float],
    method: str = "ordinary",
    mean: Optional[float] = None,
    degree: int = 1,
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
) -> tuple[FloatArray, FloatArray]:
    """3-D kriging at arbitrary target locations. Returns
    ``(predictions, variances)``; failed targets yield NaN."""

def indicator_kriging(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    cutoffs: Sequence[float],
    target_x: Sequence[float],
    target_y: Sequence[float],
    max_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    n_lags: int = 15,
    max_dist: Optional[float] = None,
    fit: str = "spherical,exponential",
    ltail: str = "linear",
    utail: str = "linear",
    tail_min: Optional[float] = None,
    tail_max: Optional[float] = None,
    mik: bool = False,
    ordinary: bool = False,
) -> dict[str, Any]:
    """Indicator kriging at arbitrary target locations. Returns a dict with
    ``ccdf`` (``n_targets x n_cutoffs`` array), ``e_type`` and
    ``cond_var``. ``fit`` is a comma-separated list of candidate variogram
    families for the per-cutoff auto-fit."""
