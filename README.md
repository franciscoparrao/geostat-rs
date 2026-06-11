# geostat-rs

Geostatistics engine in pure Rust: variography, kriging and sequential
Gaussian simulation. A modern, single-binary take on the GSLIB / gstat
feature set, with deterministic stochastic simulation and parallel
prediction out of the box.

## Crates

| Crate | Description |
|---|---|
| `geostat-core` | Library: no I/O, no heavy dependencies. `parallel` feature (rayon, default on) — disable for wasm32. |
| `geostat-cli` | `geostat` command-line tool: CSV in, CSV/JSON out. |
| `geostat-python` | Python module `geostat_rs` (PyO3, abi3 ≥ 3.9). Build with maturin. |
| `geostat-wasm` | WebAssembly bindings (wasm-bindgen); demo in `examples/wasm-demo/`. |

## Features (v0.3)

- **Experimental variograms** — omnidirectional, directional (azimuth +
  angular tolerance, gstat/GSLIB convention) and cross-variograms.
- **Theoretical models** — spherical, exponential, Gaussian, Matérn
  (ν = 3/2, 5/2), nested structures plus nugget, with optional geometric
  anisotropy per structure (major-axis azimuth + ratio).
- **Model fitting** — weighted least squares (`N_j / h_j²` weights,
  gstat's default) via Nelder–Mead; automatic best-family selection;
  2-variable LMC fitting with PSD projection.
- **Kriging** — simple, ordinary, universal (polynomial drift) and
  **external drift** (KED); **ordinary co-kriging** under a linear model
  of coregionalization; **block kriging** with explicit discretization;
  kd-tree moving neighborhoods; parallel over targets; variance maps.
- **Validation** — leave-one-out cross-validation (ME, MAE, RMSE, MSDR),
  with or without external drift.
- **Simulation** — conditional sequential **Gaussian** simulation
  (normal-score transform) and sequential **indicator** simulation
  (GSLIB-style ccdf with order-relation corrections), both with a
  deterministic, platform-independent RNG (xoshiro256++) and incremental
  bucket-grid neighbor search: same seed, same realizations, anywhere.
- **Benchmarks** — criterion suite (`cargo bench -p geostat-core`).
- **Bindings** — Python (`import geostat_rs`: variography, kriging, CV,
  SGS, SIS) and WebAssembly (browser demo in `examples/wasm-demo/`).

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

- v0.1: ✅ variography, OK/UK/SK kriging, LOO CV, SGS. Validated against
  gstat at machine precision (Meuse, Walker Lake) — see `validation/`.
- v0.2: ✅ co-kriging (LMC), KED, SIS, model anisotropy, kd-tree/bucket
  neighbor search, criterion benchmarks. KED/anisotropy/co-kriging also
  validated against gstat at machine precision.
- v0.3: ✅ Python bindings (PyO3, bit-identical with the CLI), WASM
  bindings + browser demo, block kriging (validated vs gstat at machine
  precision, including the nugget-free C̄(B,B) convention).
- Next (v0.4): 3-D support, heterotopic co-kriging, paper draft
  (Mathematical Geosciences).

## Python quickstart

```sh
pip install maturin
maturin develop -m crates/geostat-python/Cargo.toml --release
```

```python
import geostat_rs as gs

model = gs.fit_variogram(x, y, z, n_lags=15, max_dist=1500.0)
pred, var = gs.krige_grid(x, y, z, model, bbox=(0, 0, 100, 100), nx=50, ny=50,
                          max_neighbors=32)
sims = gs.sgs(x, y, z, model_ns, bbox=(0, 0, 100, 100), nx=50, ny=50,
              n_realizations=100, seed=42)
```

## License

MIT OR Apache-2.0
