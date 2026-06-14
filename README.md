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

## Features (v0.4)

- **2-D and 3-D** — the engine is generic over dimension (`PointSet<2>` /
  `PointSet<3>`); variography, kriging, CV and SGS all run in 3-D through
  the same validated code paths.
- **Experimental variograms** — omnidirectional, directional (azimuth +
  dip cone in 3-D, gstat/GSLIB convention) and cross-variograms.
- **Theoretical models** — spherical, exponential, Gaussian, Matérn
  (ν = 3/2, 5/2), nested structures plus nugget, with geometric
  anisotropy per structure (azimuth + horizontal ratio + vertical ratio).
- **Model fitting** — weighted least squares (`N_j / h_j²` weights,
  gstat's default) via Nelder–Mead; automatic best-family selection;
  2-variable LMC fitting with PSD projection.
- **Kriging** — simple, ordinary, universal (polynomial drift) and
  **external drift** (KED); **ordinary co-kriging** (collocated or
  **heterotopic**) under a linear model of coregionalization; **block
  kriging** with explicit discretization; standalone **indicator
  kriging** (local ccdf, E-type estimate, conditional variance);
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
  SGS, SIS, IK, all with 3-D variants) and WebAssembly (browser demo in
  `examples/wasm-demo/`).

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
- Next: paper draft (Mathematical Geosciences); possible block co-kriging
  and trans-Gaussian kriging.

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
```

## License

MIT OR Apache-2.0
