# geostat-rs

[![CI](https://github.com/franciscoparrao/geostat-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/franciscoparrao/geostat-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Geostatistics engine in pure Rust: variography, kriging and sequential
Gaussian/indicator simulation, plus a scalable Vecchia approximation for
large point sets. A modern, single-binary take on the GSLIB / gstat
feature set, with deterministic stochastic simulation and parallel
prediction out of the box, validated against gstat at machine precision.

Not yet published to crates.io/PyPI — build from source (see below).

## Crates

| Crate | Description |
|---|---|
| `geostat-core` | Library: no I/O, no heavy dependencies. `parallel` feature (rayon, default on) — disable for wasm32. |
| `geostat-cli` | `geostat` command-line tool: CSV in, CSV/JSON out. |
| `geostat-python` | Python module `geostat_rs` (PyO3, abi3 ≥ 3.9). Build with maturin. |
| `geostat-wasm` | WebAssembly bindings (wasm-bindgen); demo in `examples/wasm-demo/`. |

## Features (v0.7)

- **2-D and 3-D** — the engine is generic over dimension (`PointSet<2>` /
  `PointSet<3>`); variography, kriging, CV and SGS all run in 3-D through
  the same validated code paths.
- **Experimental variograms** — omnidirectional, directional (azimuth +
  dip cone in 3-D, gstat/GSLIB convention) and cross-variograms.
- **Theoretical models** — spherical, exponential, Gaussian, Matérn with
  **continuous ν** (Bessel-quadrature evaluation; ν = 3/2, 5/2 also have
  closed forms), circular, stable (power-exponential), hole-effect and
  wave (cardinal-sine), and **Power** (IRF-0, unbounded, kriged directly in
  semivariogram form). Nested multi-structure models plus nugget, with
  geometric anisotropy per structure — full 3-D rotation
  (azimuth/dip/rake, GSLIB `setrot`) and zonal anisotropy.
- **Model fitting** — weighted least squares via Nelder–Mead
  (log-parametrized, multi-start), with selectable weight schemes
  (`N_j/h_j²` gstat default, `N_j`, Cressie `N_j/γ(h_j)²`, OLS); automatic
  best-family selection; multi-structure nesting; joint `ν`/`α` fitting for
  Matérn/stable; geometric-anisotropy auto-fit; N-variable-capable LMC
  prediction (2-variable iterated Goulard–Voltz fit) with PSD projection.
- **Kriging** — simple, ordinary, universal (polynomial drift) and
  **external drift** (KED); **ordinary co-kriging** (collocated or
  **heterotopic**) and **collocated cokriging** (MM1/MM2, Journel 1999,
  core-only for now) under a linear model of coregionalization; **block
  kriging** and **block co-kriging** with explicit discretization;
  **lognormal kriging** (unbiased back-transform); **median/ordinary
  indicator kriging** (local ccdf, E-type estimate, conditional variance),
  with **Markov-Bayes** calibration of soft secondary data (core-only);
  **regression kriging** (a trend fitted separately — built-in OLS or any
  external/ML model — plus kriging of its residuals); per-datum
  **measurement error** (gstat `Err`); octant search (GSLIB `noct`) and
  `min_neighbors` (`ndmin`); kd-tree moving neighborhoods; parallel over
  targets; variance maps.
- **Vecchia approximation** — for point sets too large for exact kriging:
  O(n log n) maxmin ordering, likelihood-based maximum-likelihood/REML
  fitting (including external-drift REML and joint Matérn-`ν` MLE),
  Guinness (2018) likelihood grouping, and joint prediction (Katzfuss &
  Guinness 2021) that stays consistent across targets as `m` grows.
- **Simulation** — conditional sequential **Gaussian** simulation
  (normal-score transform, GSLIB-style tail extrapolation, cell
  declustering + weighted normal scores, multiple-grid path, separate
  data/simulated-node neighbor quotas) and sequential **indicator**
  simulation (ccdf with order-relation corrections), both with a
  deterministic, platform-independent RNG (xoshiro256++) and incremental
  bucket-grid neighbor search: same seed, same realizations, anywhere.
- **Mathematical interpolators** — inverse-distance weighting and
  k-nearest-neighbor averaging (k = 1 is nearest-neighbor / Voronoi), as
  assumption-light baselines for fair method comparison.
- **Method comparison** — a leave-one-out harness that ranks ordinary
  kriging, IDW, k-NN and NN by VEcv on the same data (`geostat compare` /
  `compare_methods`), in the spirit of Li (2021): no method dominates, so
  compare by predictive accuracy.
- **Hyperparameter tuning by accuracy** — choose the IDW power, the k-NN `k`
  or the kriging neighborhood size by maximizing leave-one-out VEcv
  (`geostat tune` / `tune_idw_power`, `tune_knn_k`, `tune_kriging_neighbors`),
  i.e. by predictive accuracy rather than by a model fit.
- **GeoPackage I/O** — read point feature layers straight from an OGC
  `.gpkg` and write kriging results back to one (pure Rust via bundled
  SQLite — no GDAL): any CLI subcommand that only needs x/y/value accepts a
  `.gpkg` input (drift/error/detrend covariate columns and 3-D mode still
  require a CSV — they fail with a clear error on `.gpkg` for now),
  `geostat gpkg-info` lists its layers, and `geostat krige -o out.gpkg`
  writes a point layer (prediction + variance) — or, with `--raster`, a
  single-band **2D-gridded-coverage** raster (16-bit PNG tile, values
  preserved via scale/offset) — with a recorded CRS, readable by QGIS/GDAL.
- **Validation** — leave-one-out, k-fold and **spatial block**
  cross-validation, with error measures (ME, MAE, MSE, RMSE, MSDR),
  scale-free relative measures (RME, RMAE, RRMSE), predictive-accuracy
  measures **VEcv** (Li 2016) and **E₁** (Legates–McCabe), and
  **Deutsch (1997) accuracy plots** for checking whether kriging variances
  are well calibrated, with or without external drift.
- **Benchmarks** — criterion suite (`cargo bench -p geostat-core`);
  proptest property tests and `cargo-fuzz` targets for parser robustness.
- **Bindings** — Python (`import geostat_rs`: variography, kriging, CV,
  SGS, SIS, IK, all with 3-D variants; zero-copy numpy outputs, GIL
  released during computation, `.pyi` type stubs) and WebAssembly (browser
  demo in `examples/wasm-demo/`).

## Build

```sh
cargo build --release         # binary at target/release/geostat
cargo test --workspace        # unit + integration tests
```

## CLI quickstart

Input is a CSV with named columns (defaults: `x`, `y`, `z`).

```sh
# 1. Experimental variogram + automatic model fit
geostat variogram -i meuse.csv --value-col zinc \
    --fit best --model-out model.json -o vario.csv

# 2. Leave-one-out cross-validation of the fitted model
geostat cv -i meuse.csv --value-col zinc -m model.json --max-neighbors 32

# 3. Ordinary kriging onto a 100x100 grid (+ kriging variance)
geostat krige -i meuse.csv --value-col zinc -m model.json \
    --nx 100 --ny 100 --max-neighbors 32 -o kriged.csv

# 4. Conditional SGS, 100 realizations, reproducible
geostat sgs -i meuse.csv --value-col zinc \
    --nx 100 --ny 100 -n 100 --seed 42 -o sims.csv

# 5. Kriging with external drift (covariates in data and targets CSVs)
geostat krige -i meuse.csv --value-col lzinc -m model.json \
    --drift-cols sdist --targets grid_with_sdist.csv -o ked.csv

# 6. Ordinary co-kriging with a collocated secondary variable
geostat cokrige -i meuse.csv --value-col lzinc --secondary-col llead \
    --nx 100 --ny 100 -o cokriged.csv     # LMC auto-fitted (or --lmc lmc.json)

# 7. Sequential indicator simulation at the quartiles
geostat sis -i meuse.csv --value-col zinc --quantiles 0.25,0.5,0.75 \
    --nx 100 --ny 100 -n 50 --seed 42 -o sis.csv

# 8. Indicator kriging: local ccdf, E-type estimate, conditional variance
geostat ik -i meuse.csv --value-col zinc --quantiles 0.25,0.5,0.75 \
    --nx 100 --ny 100 -o ik.csv

# 9. 3-D ordinary kriging (z column + explicit targets CSV with x,y,z)
geostat krige -i drillholes.csv --z-col z --value-col grade -m model3d.json \
    --targets blocks.csv -o kriged3d.csv

# 10. Heterotopic co-kriging (secondary at its own locations; needs --lmc)
geostat cokrige -i primary.csv --value-col lzinc --secondary-col llead \
    --secondary-input secondary.csv --lmc lmc.json \
    --nx 100 --ny 100 -o cokriged.csv

# 11. Regression kriging: OLS trend on covariates + kriging of residuals
#     (residual variogram fitted automatically; targets carry the covariates)
geostat rk -i meuse.csv --value-col lzinc --covar-cols sdist \
    --targets grid_with_sdist.csv -o rk.csv

# 12. Compare methods (OK / IDW / k-NN / NN) by leave-one-out VEcv
geostat compare -i meuse.csv --value-col zinc --max-neighbors 32 --knn-k 8

# 13. Tune a hyperparameter by leave-one-out VEcv (idw | knn | ok)
geostat tune -i meuse.csv --value-col zinc --method idw

# 14. GeoPackage I/O: read points from a .gpkg, write kriging to one
geostat gpkg-info -i meuse.gpkg                            # list layers
geostat cv -i meuse.gpkg --value-col lzinc -m model.json   # any subcommand reads .gpkg
geostat krige -i meuse.gpkg --value-col lzinc -m model.json \
    --nx 100 --ny 100 --srs 28992 -o kriged.gpkg           # write a .gpkg point layer
geostat krige -i meuse.gpkg --value-col lzinc -m model.json \
    --nx 200 --ny 200 --srs 28992 --raster -o kriged.gpkg  # write a single-band raster

# 15. Vecchia: scalable ML covariance fitting + prediction for large n
#     (O(n log n) plan; --trend for REML under a spatial trend)
geostat variogram -i drillholes.csv --value-col grade \
    --mle --cond 20 --model-out model_ml.json
geostat krige -i drillholes.csv --value-col grade -m model_ml.json \
    --vecchia 20 --nx 200 --ny 200 -o kriged_vecchia.csv

# 16. Spatial block CV + Deutsch accuracy plot (honest error under autocorrelation)
geostat cv -i meuse.csv --value-col zinc -m model.json --blocks 4,4 --accuracy
```

Other useful flags: `--azimuth/--dip/--tolerance` (directional variograms,
dip for 3-D), `--method simple|ordinary|universal --degree 2` (kriging
flavor), `--block w,h --block-discr nx,ny` (block kriging),
`--bbox xmin,ymin,xmax,ymax --res <cell>` (grid control).

A fitted model is plain JSON and can be edited by hand:

```json
{
  "nugget": 0.05,
  "structures": [{ "kind": "spherical", "sill": 0.59, "range": 897.0 }]
}
```

## Library quickstart

```rust
use geostat_core::{
    experimental_variogram, fit_best, Kriging, KrigingConfig, ModelKind,
    PointSet, VariogramConfig,
};

let data = PointSet::from_xyz(&x, &y, &z)?;
let ev = experimental_variogram(&data, &VariogramConfig {
    n_lags: 15, max_dist: 1500.0, direction: None,
})?;
let fit = fit_best(&ev, &ModelKind::ALL)?;
let kriging = Kriging::new(&data, &fit.model, KrigingConfig::default())?;
let est = kriging.predict([179_000.0, 330_000.0])?;
println!("prediction {} ± {}", est.value, est.variance.sqrt());
```

## Validation against gstat

Every deterministic method is cross-checked against **gstat** (R) at
machine precision on the Meuse and Walker Lake datasets; SGS is validated
distributionally. The full harness and result tables live in
[`validation/`](validation/README.md) — variography, OK/UK/SK/KED/co-kriging
(collocated and heterotopic), block kriging, 3-D kriging/CV and indicator
kriging, plus the SGS ensemble check.

Conventions intentionally match gstat where it matters: semivariance
estimator, `N_j/h_j²` fit weights, azimuth measured clockwise from north,
right-closed lag bins, nugget-free within-block covariance. The Matérn
parameterization is Rasmussen & Williams (`√(2ν)h/ρ` scaling), so ranges
are comparable across families.

## Roadmap

- v0.1: ✅ variography, OK/UK/SK kriging, LOO CV, SGS. Validated against
  gstat at machine precision (Meuse, Walker Lake) — see `validation/`.
- v0.2: ✅ co-kriging (LMC), KED, SIS, model anisotropy, kd-tree/bucket
  neighbor search, criterion benchmarks. KED/anisotropy/co-kriging also
  validated against gstat at machine precision.
- v0.3: ✅ Python bindings (PyO3, bit-identical with the CLI), WASM
  bindings + browser demo, block kriging (validated vs gstat at machine
  precision, including the nugget-free C̄(B,B) convention).
- v0.4: ✅ 3-D support (const-generic core), heterotopic co-kriging,
  standalone indicator kriging — all validated against gstat at machine
  precision; 3-D and IK exposed in the Python bindings.
- v0.5: ✅ lognormal (trans-Gaussian) kriging and block co-kriging.
  Block co-kriging matches gstat to machine precision; simple lognormal
  kriging matches gstat's SK back-transform to ~1e-9 (ordinary follows the
  Journel & Huijbregts formula).
- v0.6: ✅ VEcv/E₁ and relative error measures in cross-validation (parity
  with Li's spm::pred.acc); ✅ regression kriging (separate trend + residual
  kriging), the bridge to an ML trend engine; ✅ IDW/k-NN/NN baselines + a
  VEcv method-comparison harness; ✅ hyperparameter tuning by predictive
  accuracy (IDW power, k-NN k, kriging neighborhood); ✅ GeoPackage I/O
  (point reading + point-layer writing + single-band raster /
  2D-gridded-coverage output).
- v0.7: ✅ Vecchia approximation (O(n log n) plan, MLE/REML, Guinness
  grouping, joint prediction) for large point sets; ✅ Matérn with
  continuous ν, plus circular/stable/hole/wave/Power families, full 3-D
  rotation and zonal anisotropy; ✅ cell declustering + weighted normal
  scores; ✅ GSLIB tail extrapolation for SGS/SIS/IK back-transforms;
  ✅ median/ordinary indicator kriging, collocated cokriging (MM1/MM2) and
  Markov-Bayes soft-data calibration; ✅ per-datum measurement error,
  octant search; ✅ spatial block CV and Deutsch accuracy plots;
  ✅ public `Covariance` trait, zero-copy numpy bindings, proptest/fuzz
  coverage. See `docs/AUDIT-2026-07.md` and `docs/AUDIT-2026-07-v2.md` for
  the full audit trail behind this release.
- ML+geostatistics hybrids: regression kriging accepts an external trend, so
  an ML model supplies the mean and geostat-rs kriges the residuals. See
  `examples/hybrid_smelt_rk.py` for an RFOK-style hybrid built entirely from
  the author's Rust engines — a Smelt random-forest trend + residual kriging
  here — scored by VEcv.
- Next: paper draft (Mathematical Geosciences); publish to crates.io/PyPI.

## Python quickstart

```sh
pip install maturin
maturin develop -m crates/geostat-python/Cargo.toml --release
```

```python
import geostat_rs as gs

# 2-D
model = gs.fit_variogram(x, y, vals, n_lags=15, max_dist=1500.0)
pred, var = gs.krige_grid(x, y, vals, model, bbox=(0, 0, 100, 100),
                          nx=50, ny=50, max_neighbors=32)
sims = gs.sgs(x, y, vals, model_ns, bbox=(0, 0, 100, 100), nx=50, ny=50,
              n_realizations=100, seed=42)

# 3-D (drillhole-style data)
m3 = gs.fit_variogram_3d(x, y, z, grade, n_lags=12, max_dist=200.0)
pred, var = gs.krige_3d(x, y, z, grade, m3, bx, by, bz, max_neighbors=24)

# Indicator kriging → local ccdf + E-type estimate
ik = gs.indicator_kriging(x, y, vals, cutoffs, tx, ty)
ik["ccdf"], ik["e_type"], ik["cond_var"]

# Regression kriging: OLS trend on covariates + residual kriging.
# (Or pass trend_at_data / trend_at_targets from any external/ML model.)
rk = gs.regression_kriging(x, y, vals, covars, tx, ty, target_covars)
rk["prediction"], rk["variance"], rk["trend_coef"]

# Ordinary co-kriging with a correlated secondary (auto-fitted LMC;
# collocated or heterotopic; `ridge` stabilizes the system).
pred, var = gs.co_kriging(px, py, pv, sx, sy, sv, tx, ty)

# Baselines + method comparison by leave-one-out VEcv.
idw_pred = gs.idw(x, y, vals, tx, ty, power=2.0, max_neighbors=16)
ranking = gs.compare_methods(x, y, vals, max_neighbors=32, knn_k=8)
# {"ordinary_kriging": {"rmse":..., "vecv":...}, "idw": {...}, ...}

# Tune a hyperparameter by predictive accuracy (VEcv).
best = gs.tune_idw_power(x, y, vals)          # {"best":..., "best_vecv":..., "trace":[...]}
gs.tune_knn_k(x, y, vals); gs.tune_kriging_neighbors(x, y, vals)
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
