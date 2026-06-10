# Numerical parity validation against gstat (R)

Cross-check of geostat-rs against **gstat** on the classic Meuse dataset
(`log(zinc)`, 155 points), as required before trusting the engine for
publication work.

## Results (2026-06-10, gstat 2.1.4 / sp 2.2.0)

| Check | Max difference | Verdict |
|---|---|---|
| Experimental variogram: pair counts (15 lags) | 0 | exact |
| Experimental variogram: mean lag distance | 3.0e-12 | machine precision |
| Experimental variogram: gamma (relative) | 1.6e-15 | machine precision |
| Fitted Sph model: nugget / sill / range (relative) | ~1e-6 | independent optimizers agree |
| OK predictions, meuse.grid (3103 cells) | 1.5e-12 | machine precision |
| OK kriging variances | 7.3e-13 | machine precision |
| LOO CV predictions (155 points) | 9.7e-13 | machine precision |
| LOO CV variances | 5.7e-13 | machine precision |

Conventions aligned with gstat: semivariance estimator, right-closed lag
intervals `((b-1)w, bw]`, WLS fit weights `N_j/h_j²` (fit.method = 7),
covariance-form ordinary kriging with global neighborhood.

## Reproduce

```sh
cargo build --release
Rscript validation/gstat_reference.R

BIN=target/release/geostat
$BIN variogram -i validation/out/meuse_lzinc.csv --value-col lzinc \
    --n-lags 15 --max-dist 1500 \
    -o validation/out/rust_vario.csv \
    --fit spherical --model-out validation/out/rust_model.json
$BIN krige -i validation/out/meuse_lzinc.csv --value-col lzinc \
    -m validation/out/gstat_model.json \
    --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
    -o validation/out/rust_krige.csv
$BIN cv -i validation/out/meuse_lzinc.csv --value-col lzinc \
    -m validation/out/gstat_model.json -o validation/out/rust_cv.csv

python3 validation/compare.py
```

`validation/out/` is regenerated on each run and not tracked by git.

## Notes

- The kriging/CV comparison uses **gstat's fitted model on both sides**, so
  it isolates the kriging engine from the fit. The fit comparison is
  reported separately (different optimizers; agreement to ~1e-6 relative
  on identical bins).
- Pending for the paper: same exercise on Walker Lake, and SGS ensemble
  statistics vs gstat's conditional simulation (`nsim`), which can only be
  compared distributionally (different RNGs).
