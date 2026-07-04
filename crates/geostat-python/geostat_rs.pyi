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
) -> tuple[list[float], list[float], list[int]]:
    """Experimental semivariogram. Returns ``(h, gamma, n_pairs)`` lists;
    empty bins carry NaN gamma."""

def variogram_map(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    lag_width: float = 1.0,
) -> dict[str, Any]:
    """2-D variogram map (lag-space semivariance surface). Returns a dict
    with ``size``, ``lag_width``, and flat row-major lists ``hx``, ``hy``,
    ``gamma``, ``n_pairs``."""

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
) -> tuple[FloatArray, FloatArray]:
    """Kriging at arbitrary target locations. Returns
    ``(predictions, variances)``; failed targets yield NaN."""

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
) -> tuple[FloatArray, FloatArray]:
    """Kriging over a regular grid. Returns ``(predictions, variances)``,
    row-major with y increasing."""

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
    ``rmse``, ``msdr``, ``vecv``, ``e1``, ``predicted`` and ``variance``."""

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
    lower_tail: str = "none",
    upper_tail: str = "none",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    decluster_cell: Optional[float] = None,
    max_node_neighbors: Optional[int] = None,
    multigrid: int = 0,
) -> FloatArray:
    """Conditional sequential Gaussian simulation. Returns an
    ``n_realizations x n_cells`` array, one row per realization."""

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
    ltail: str = "linear",
    utail: str = "linear",
    tail_min: Optional[float] = None,
    tail_max: Optional[float] = None,
    mik: bool = False,
    ordinary: bool = False,
) -> FloatArray:
    """Conditional sequential indicator simulation. Returns an
    ``n_realizations x n_cells`` array, one row per realization."""

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
    ltail: str = "linear",
    utail: str = "linear",
    tail_min: Optional[float] = None,
    tail_max: Optional[float] = None,
    mik: bool = False,
    ordinary: bool = False,
) -> dict[str, Any]:
    """Indicator kriging at arbitrary target locations. Returns a dict with
    ``ccdf`` (``n_targets x n_cutoffs`` array), ``e_type`` and
    ``cond_var``."""
