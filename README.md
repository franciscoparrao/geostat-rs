# geostat-rs

Geostatistics engine in pure Rust: variography, kriging and sequential
Gaussian simulation. A modern, single-binary take on the GSLIB / gstat
feature set, with deterministic stochastic simulation and parallel
prediction out of the box.

## Crates

| Crate | Description |
|---|---|
| `geostat-core` | Library: no I/O, no heavy dependencies. Targets native (Rayon), Python (PyO3, planned) and WASM (planned). |
| `geostat-cli` | `geostat` command-line tool: CSV in, CSV/JSON out. |

## Features (v0.1 MVP)

- **Experimental variograms** — omnidirectional and directional
  (azimuth + angular tolerance, gstat/GSLIB convention).
- **Theoretical models** — spherical, exponential, Gaussian, Matérn
  (ν = 3/2, 5/2), nested structures plus nugget.
- **Model fitting** — weighted least squares (`N_j / h_j²` weights,
  gstat's default) via Nelder–Mead; automatic best-family selection.
- **Kriging** — simple, ordinary and universal (linear/quadratic drift),
  global or moving neighborhood (k-nearest, search radius), parallel
  over prediction targets, kriging variance maps.
- **Validation** — leave-one-out cross-validation: ME, MAE, RMSE, MSDR.
- **Simulation** — conditional sequential Gaussian simulation with
  normal-score transform and a deterministic, platform-independent RNG
  (xoshiro256++): same seed, same realizations, on any machine.

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
```

Other useful flags: `--azimuth/--tolerance` (directional variograms),
`--method simple|ordinary|universal --degree 2` (kriging flavor),
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

The numerical cross-check against **gstat** (R) on the Meuse and Walker
Lake datasets is part of the v0.1 roadmap. To export Meuse from R:

```r
library(sp); data(meuse)
write.csv(meuse[, c("x", "y", "zinc")], "meuse.csv", row.names = FALSE)
```

Conventions intentionally match gstat where it matters: semivariance
estimator, `N_j/h_j²` fit weights, azimuth measured clockwise from north.
Note the Matérn parameterization is Rasmussen & Williams (`√(2ν)h/ρ`
scaling), so ranges are comparable across families.

## Roadmap

- v0.1: ✅ variography, OK/UK/SK kriging, LOO CV, SGS — paridad numérica
  con gstat pendiente.
- v0.2: co-kriging, kriging with external drift, sequential indicator
  simulation (SIS), anisotropy in models (not just experimental),
  kd-tree neighbor search, `criterion` benchmarks.
- Targets: Python bindings (PyO3), WASM demo.

## License

MIT OR Apache-2.0
