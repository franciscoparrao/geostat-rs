# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
before 0.7.0 predate this file and are reconstructed from commit history.

## [Unreleased]

### Added
- Collocated cokriging (MM1/MM2, Journel 1999) exposed in the CLI
  (`geostat collocated-cokrige`) and Python (`collocated_cokriging`,
  `collocated_stats`) — the core (`crates/geostat-core/src/collocated.rs`)
  has had this since the Fase 6 #17 session, but it was never wired up to
  either surface. Takes an explicit primary variogram model (and, for MM2,
  the secondary's own model) plus `mean1/mean2/rho12/sigma1/sigma2` (either
  supplied directly or auto-estimated from collocated sample pairs via the
  new `estimate_collocated_stats` exposure); predicts at explicit targets
  carrying one secondary value each (e.g. sampled from an exhaustive raster
  with `gpkg-sample`), since that per-target secondary value is the whole
  point of collocated cokriging. New `CollocatedCokriging::predict_many`
  in the core (parallel batch prediction, NaN on a per-target failure
  instead of aborting, matching `Kriging::predict_many`'s convention) backs
  both surfaces.
- Truncated Gaussian simulation (TGS) for ordered categorical/facies data
  (`crates/geostat-core/src/tgs.rs`; `geostat tgs` in the CLI, `tgs()` in
  Python) — one underlying Gaussian field, simulated via the same
  sequential-conditioning engine as SGS (now shared through a new
  transform-agnostic `simulate_gaussian_path` extracted from
  `simulation.rs`'s per-realization loop), truncated into ordered
  categories at thresholds derived from global proportions via the inverse
  normal CDF (`tgs_thresholds`/`tgs_classify`; hard categorical data is
  converted to a pseudo-Gaussian conditioning value via
  `category_to_pseudo_gaussian`). `TgsConfig`/`CategoricalData` follow this
  project's established `#[non_exhaustive]`-config and dimension-generic
  (`const D: usize`) conventions. The underlying-field variogram is always
  caller-supplied (never auto-fitted: TGS conditions on only a handful of
  discrete pseudo-Gaussian levels, too few for a direct experimental fit —
  GSLIB practice calibrates it against target facies indicator variograms
  instead, an external step not automated here). Validated by
  self-consistency (no gstat/GSLIB TGS to cross-check against): ensemble
  category proportions track the input proportions, hard data is honored
  exactly at conditioning locations, thresholds match `inv_norm_cdf` by
  hand. Full plurigaussian simulation (2+ correlated fields, a flexible
  2-D truncation rule for non-ordered facies) is explicitly out of scope
  for this pass — see the module docs.

## [0.8.0] — 2026-07-10

Fase 6 operational-gap closing (`docs/AUDIT-2026-07-v2.md` §7) plus Fase 7,
the third audit pass (`docs/AUDIT-2026-07-v3.md`): 2 HIGH + 12 MEDIUM findings,
each with a dedicated regression test. No functional regressions; gstat
parity on Meuse re-confirmed after every fix (gamma 2.07e-15, kriging
1.5e-12, CV 9.7e-13).

### Added
- Anisotropic (rotated-ellipsoid) search neighborhoods for kriging (GSLIB
  `kt3d` `sang1/sang2/sang3/sanis1/sanis2`): `KrigingConfig::anisotropic_search`,
  CLI `--search-azimuth/--search-ratio/--search-ratio-z/--search-dip/--search-rake`,
  Python `krige`/`krige_grid` `search_azimuth=`/etc. Octant classification
  follows the rotated frame when set.
- SIS brought to parity with SGS: reusable Cholesky workspace for simple
  indicator kriging (was a fresh `Array2` allocation + LU solve per
  node/cutoff even though the simple-IK system is SPD), separate
  data/simulated-node neighbor quotas (`max_node_neighbors`, GSLIB
  `nodmax`), a multiple-grid path (`multigrid`, GSLIB `nmult`),
  declustering-weighted global cutoff proportions (`decluster_weights`,
  previously ignored entirely), and a 3-D grid entry point
  (`sequential_indicator_simulation_3d`, previously SGS-only). Exposed in
  the CLI (`sis --declus/--nodmax/--multigrid`) and Python (`sis(...)`
  gains `decluster_cell=`/`max_node_neighbors=`/`multigrid=`).
- Robust/alternative point-pair variogram estimators (`EstimatorKind`:
  Cressie-Hawkins, Dowd, madogram, alongside the existing Matheron
  default) via `experimental_variogram_robust`, CLI `variogram --estimator`
  and Python `experimental_variogram(estimator=...)` — the classical
  mean-squared-difference estimator lets a single outlier pair dominate a
  lag bin; Dowd's median-based estimator is immune to it entirely. Also:
  the variogram cloud (`variogram_cloud`, one unbinned point per pair, for
  tracing an outlier bin back to its locations), an ergodic correlogram
  derived from an experimental variogram (`correlogram`), and a
  `coincident_pairs` count on `ExperimentalVariogram` (previously silently
  dropped). Non-ergodic (per-lag head/tail) correlograms and relative
  variograms are explicitly out of scope for this pass — see
  `docs/AUDIT-2026-07-v2.md` §4.
- Analytical gradients for the Vecchia log-likelihood w.r.t. `(nugget,
  psill, range)` for the covariance families with a closed-form range
  derivative (Spherical, Exponential, Gaussian, Matérn 3/2, Matérn 5/2),
  propagated through each point's local GLS system with one extra `O(m^2)`
  triangular solve per parameter on top of the existing `O(m^3)`
  factorization. A new `bfgs`/`bfgs_multistart` quasi-Newton optimizer
  (`optim.rs`) uses them to fit `vecchia_mle`/`vecchia_mle_grouped` --
  orders of magnitude fewer likelihood evaluations than the previous
  Nelder-Mead multistart (measured: ~40x wall-clock on a 60-point/m=12
  fit) with an identical fitted optimum. Kinds without a closed-form
  gradient, and grouped (Guinness-blocked) likelihoods, keep the
  gradient-free Nelder-Mead path unchanged. `vecchia_param_se` reuses the
  same analytical gradient for a semi-analytical Hessian (finite-
  differencing the exact gradient once, instead of double-differencing
  the raw log-likelihood) for the same covariance families, addressing
  the "Hessiano por diferencias finitas frágil" note in
  AUDIT-2026-07-v2.md §5.1. Full Fisher scoring, a custom `erfc`, a
  small-*x* Bessel branch, and configurable rcond/jitter policy remain
  out of scope for this pass.
- Unified cross-validation, previously `Kriging`-only:
  `leave_one_out_cokriging`, `leave_one_out_collocated`,
  `leave_one_out_lognormal` (with the correct back-transformed lognormal
  predictive variance, not just log-space variance reused verbatim), and
  `leave_one_out_indicator` (returns an `IkCvResult` with a Ranked
  Probability Score over the held-out ccdfs, since IK's whole point is a
  distribution, not a point estimate RMSE-family measures assume). Also:
  `accuracy_plot_ccdf` (Deutsch 1997's original ccdf-based formulation,
  reusing `sis`'s ccdf-interpolation as a deterministic quantile lookup
  instead of `accuracy_plot`'s Gaussian-interval approximation), and
  `realization_variogram_qc` (SGS/SIS ensemble variogram-reproduction
  check, promoting a check every simulation workflow already does
  informally to a library API). k-fold/block CV accepting external
  drift/measurement error, and CLI/Python exposure of the new CV
  functions, remain out of scope for this pass (core-only, matching the
  precedent already set for collocated cokriging/Markov-Bayes).

### Changed
- `KrigingConfig` is now `#[non_exhaustive]` (was already the case for
  `SgsConfig`/`SisConfig`/etc.).

### Fixed
- `BucketGrid::new` aborted the process (OOM, not a catchable `Err`) when
  all points were collinear along one axis (e.g. a single-bench drillhole
  with constant `z`) — reachable from `vecchia_plan`/`vecchia_predict`/
  `sgs_at`/`sis_at`. Degenerate axes now get `n[d]=1` instead of an
  astronomically small cell size, plus a defensive cap on total cell count.
- The experimental variogram — and any auto-fit pipeline built on it
  (`sgs`/`sis`/`krige --fit`) — was not bit-reproducible across machines
  with different thread counts: the pair-accumulation chunk count scaled
  with `rayon::current_num_threads()`, so floating-point summation order
  (and therefore `gamma`) changed with the CPU. Chunk count is now
  independent of thread count.
- `MATERN_NU_MAX` (50) exceeded the Bessel quadrature's validated domain
  (ν ≤ 15), silently producing negative `gamma` (a non-PD covariance) for
  ν between 15 and 50; lowered to 15.
- Collocated cokriging's system could be silently non-PSD when the sample
  secondary/primary variance didn't match the fitted model's sill (the v2
  fix only corrected the cross-covariance, not the primary block); the
  primary block is now standardized to the correlogram, guaranteeing PSD
  by construction (ρ² ≤ 1).
- Dowd's robust variogram estimator used the wrong normalizing constant
  (0.4529 instead of 1/Φ⁻¹(0.75)² = 0.454936), a systematic +0.5% bias
  confirmed both analytically and by Monte Carlo.
- The 3-D `Circular` dimensional guard from the v2 audit didn't reach
  `sgs_at_with_levels`, `Lmc::new`, or the per-kind Vecchia MLE/REML entry
  points; now rejected consistently everywhere `ModelKind` is
  dimension-sensitive.
- Vecchia's analytical gradient kept propagating a nonzero `d(var)` even
  when the variance clamp (near-singular neighborhoods) made the
  likelihood locally constant in `var`, corrupting BFGS curvature and the
  semi-analytical Hessian used by `vecchia_param_se`; the gradient is now
  zeroed when the clamp is active.
- `vecchia_param_se` returned `[NaN; 3]` for all three parameters
  (including the well-identified sill/range) whenever nugget < 1e-5,
  because the two-sided finite-difference step could go negative; it now
  falls back to a one-sided difference in that case.
- `bfgs` could silently return the starting point as the "optimum" with no
  convergence signal (e.g. on the Vecchia penalty plateau, where the
  gradient is exactly zero); the optimizer now tracks the best point
  visited, and Vecchia's MLE fit errors explicitly if `neg_ll >= 1e12` for
  every multistart instead of returning it unmarked.
- Anisotropic kriging search parameters (`--search-ratio`, azimuth/dip/
  rake) accepted zero/negative/non-finite values, silently producing
  neighborhoods keyed on the sign of a divide-by-zero numerator rather
  than distance; now validated finite and `> 0` in `Kriging::build`.
- CLI: `--estimator madogram` combined with `--fit` fit a variogram model
  in madogram scale — a nonlinear transform of gamma under Gaussianity —
  and wrote it out for kriging with no warning; now rejected.
- GeoPackage raster reads (`read_raster`) could panic or OOM on a corrupt
  or malicious tile matrix (unchecked `i64 -> usize` casts, unchecked
  multiplication); dimensions are now validated and multiplication is
  checked.
- `decode_point` didn't skip the embedded SRID in EWKB-flavored geometries,
  silently reading garbage coordinates (SRID bytes interpreted as part of
  the coordinate); the SRID flag is now recognized and skipped.
- CLI `krige --targets` was silently ignored in plain 2-D kriging (only
  consumed by the 3-D/external-drift branches); now rejected with a clear
  error, along with `--raster` on non-`.gpkg` output and `--blocks`+
  `--folds` both being passed to `cv`.

## [0.7.0] — 2026-07-04

Audit-driven hardening pass (`docs/AUDIT-2026-07.md`, `docs/AUDIT-2026-07-v2.md`)
and publication prep. No functional regressions vs 0.6.0; several bug fixes
and a handful of intentional, pre-1.0 API renames (see below).

### Added
- Public `Covariance<const D: usize>` trait: krige against a custom
  covariance function without going through `VariogramModel`.
- Matérn with continuous `ν` (Bessel-quadrature evaluation), plus `Circular`,
  `Stable(α)`, `Hole`, `Wave` and `Power` (IRF-0, kriged directly in
  semivariogram form) variogram families. Full 3-D rotation (`ang1/ang2/ang3`
  GSLIB `setrot`), zonal anisotropy (`ratio > 1`), joint `ν`/`α` fitting,
  multi-structure nesting (`fit_nested`), and selectable WLS weight schemes
  (`FitWeights`: `NPairs`, `Cressie`, `Ols`, `NOverHSquared`).
- Vecchia approximation: O(n log n) plan construction (maxmin + incremental
  predecessors), Guinness (2018) likelihood grouping, `vecchia_predict`
  (Katzfuss & Guinness 2021 joint prediction), REML/trend-REML and
  external-drift REML fitting, likelihood-based parameter standard errors,
  joint Matérn-`ν` MLE.
- Collocated cokriging (MM1/MM2, Journel 1999) and Markov-Bayes calibration
  of soft secondary data for indicator kriging.
- Median and ordinary indicator kriging; GSLIB-style tail extrapolation
  (`ltail`/`utail`: linear, power, hyperbolic) for SGS, SIS and indicator
  kriging back-transforms.
- Cell declustering (GSLIB `declus`) and weighted normal-score transforms.
- Per-datum measurement error (gstat `Err` parity), octant search
  (GSLIB `noct`) and `min_neighbors` (`ndmin`) for kriging.
- OLS residual variograms for universal/external-drift kriging.
- Spatial block cross-validation and Deutsch (1997) accuracy plots.
- SGS: separate data/simulated-node neighbor quotas (`nodmax`) and a
  multiple-grid path.
- 2-D variogram map (`variogram_map`) and automatic geometric-anisotropy
  fitting (`fit_anisotropic`).
- rust-numpy-backed Python outputs (zero-copy arrays) with `allow_threads`
  on every non-trivial call; `.pyi` type stubs + `py.typed` marker;
  `pyproject.toml` with dynamic (Cargo-sourced) versioning.
- proptest property tests (variogram/kriging/SIS/linalg invariants) and
  `cargo-fuzz` targets for `ModelKind` parsing and `VariogramModel` JSON.
- Multi-OS CI (test + clippy + fmt + `wasm32` check + MSRV).
- `LICENSE-MIT`, `LICENSE-APACHE`, this changelog.

### Changed
- **Breaking (pre-1.0, not yet published):** `GeostatError` and the public
  config structs (`SisConfig`, `IkConfig`, `CoKrigingConfig`, `SgsConfig`,
  `CollocatedConfig`) are now `#[non_exhaustive]` — construct via
  `Config::default()` plus field assignment rather than a full struct
  literal (adding a field is no longer a breaking change).
- **Breaking:** Python `sgs`'s tail parameters renamed `lower_tail`/
  `upper_tail` → `ltail`/`utail`, matching `sis`/`indicator_kriging` and the
  CLI (defaults unchanged).
- **Breaking:** Python `variogram_map`'s `lag_width` default changed from a
  hardcoded `1.0` to a data-driven default (a fifteenth of the bounding-box
  half-diagonal, matching the CLI); pass `lag_width=None` explicitly for the
  new behavior, same as omitting it.
- Python `sis`/`indicator_kriging` gained a `fit` parameter (comma-separated
  variogram-family spec, same syntax as the CLI's `--fit`) instead of a
  hardcoded `[Spherical, Exponential]` candidate list.
- Python `loo_cv` now includes `observed` in its result dict (previously
  undocumented-but-referenced by `accuracy_plot`'s own docstring).
- Bumped to Rust edition 2024, MSRV 1.88.

### Fixed
- 3-D block kriging dropped the `z` separation (used 2-D distances inside a
  const-generic-3-D code path).
- `fit_lmc`'s WLS fit mixed an isotropic base curve with an anisotropic
  result; replaced with iterated Goulard–Voltz (1992) and an explicit
  rejection of anisotropic templates instead.
- Ridge default diverged between the CLI (`0.0`) and Python (`1e-2`) for
  co-kriging; unified to `0.0` in the core.
- Duplicate coordinates silently produced a singular system (NaN estimates)
  in `Kriging`, `CollocatedCokriging` and `CoKriging`; now rejected
  up front with a clear error.
- Directional 3-D variograms: the experimental variogram's `dip` sign
  convention didn't match the fitted model's rotation, silently mirroring
  anisotropy end to end (`DirectionConfig`/`Anisotropy` now share one
  convention, cross-checked by a dedicated test).
- `Power` (IRF-0) models combined with measurement error or lognormal
  kriging used the wrong sign for the semivariogram-form diagonal/Lagrange
  multiplier; both combinations are now rejected explicitly instead of
  silently miscomputing.
- `vecchia_predict` was not bit-for-bit deterministic across runs (a
  `HashMap`'s per-process random iteration order fed a floating-point sum).
- `ModelKind::Circular` (a 2-D-only covariance) had no guard against 3-D use
  across every engine (kriging, Vecchia, SIS, indicator kriging, collocated
  cokriging); `Matern`'s `ν` had no upper bound, silently producing NaN past
  the point where `Γ(ν)` overflows (~171.6) — both now rejected explicitly.
- Collocated cokriging's cross-covariance used the raw covariance instead of
  a correlogram, breaking internal consistency whenever the caller-supplied
  `sigma1`/`sigma2` didn't exactly match the model's own sill.
- Markov-Bayes indicator kriging used the hard indicator's global proportion
  as the soft channel's mean instead of its own calibrated mean.
- The CLI silently routed `.gpkg` inputs through the CSV parser for
  3-D/drift/error-column reads instead of failing with a clear message.
- `vecchia_reml_drift`'s trend basis wasn't centered/scaled like the plain
  polynomial-trend path, leaving UTM-scale covariates needlessly
  ill-conditioned.

## [0.6.0] — 2026-06-15

Kriging with transport (warped kriging): a bridge to Transport Gaussian
Process marginals (Box-Cox, Yeo-Johnson, sinh-arcsinh, fitted by maximum
likelihood), latent-space kriging + Monte Carlo back-transform, anchored to
the analytic lognormal case at <1% agreement. The `tgp`/`warped_kriging`
module and its CLI subcommand were later extracted to a private crate
(2026-06); this repository has carried only `optim` (the Nelder–Mead helper
extracted alongside it) since.

## [0.5.0] — 2026-06-15

Lognormal (trans-Gaussian) kriging with the Journel & Huijbregts
back-transform, and block co-kriging.

## [0.4.0] — 2026-06-13

Core generalized to arbitrary spatial dimension (`PointSet<const D>`,
D-dimensional kd-tree and bucket grid). Heterotopic co-kriging, 3-D
anisotropy (dip/rake), 3-D polynomial drift, standalone indicator kriging
(local ccdf, E-type, conditional variance). 3-D and indicator kriging
exposed in the Python bindings.

## [0.3.0] — 2026-06-11

PyO3 Python bindings (bit-identical to the CLI) and a WebAssembly demo.
Block kriging.

## [0.2.0] — 2026-06-11

Co-kriging with a fitted linear model of coregionalization (LMC), kriging
with external drift, sequential indicator simulation, geometric
anisotropy in variogram models, a kd-tree/bucket-grid search index, and
criterion benchmarks.

## [0.1.0] — 2026-06-10

Initial release: experimental variograms (isotropic/anisotropic) with model
fitting (spherical, exponential, gaussian, Matérn 3/2 and 5/2), simple/
ordinary/universal kriging, leave-one-out cross-validation, and sequential
Gaussian simulation with a deterministic, cross-platform xoshiro256++ RNG.
Validated against gstat (R) on the Meuse and Walker Lake datasets.
