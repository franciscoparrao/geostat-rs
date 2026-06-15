# Numerical parity validation against gstat (R)

Cross-check of geostat-rs against **gstat** on the classic Meuse and
Walker Lake datasets, as required before trusting the engine for
publication work.

## Meuse results (2026-06-10, gstat 2.1.4 / sp 2.2.0)

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

## Walker Lake results (V variable, 470 points)

Deterministic parity, same harness (15 lags / cutoff 120, OK on a 26x30
grid, LOO CV, global neighborhood, gstat's fitted model on both sides):

| Check | Max difference | Verdict |
|---|---|---|
| Experimental variogram: pair counts | 0 | exact |
| Experimental variogram: gamma (relative) | 3.0e-15 | machine precision |
| Fitted Sph model: nugget / sill / range (relative) | ~2e-5 | optimizers agree |
| OK predictions / variances (780 cells) | 2.4e-12 / 2.1e-10 | machine precision |
| LOO CV predictions / variances | 5.2e-12 / 2.5e-10 | machine precision |

## SGS distributional validation (Walker Lake normal scores)

Different RNGs make realizations incomparable one-to-one, so the check is
distributional: 1000 conditional Gaussian simulations per engine (simple
kriging, mean 0, 16 neighbors; gstat `krige(..., beta = 0, nmax = 16,
nsim = 1000)` vs `geostat sgs`), compared through ensemble statistics on
780 nodes:

| Check | Result | Bound |
|---|---|---|
| Ensemble mean fields, RMSE / correlation | 0.047 / 0.998 | 0.08 / 0.98 |
| Rust ensemble mean vs theoretical SK prediction, RMSE | 0.058 (gstat: 0.067) | 0.06 |
| Ensemble std vs theoretical SK std, mean abs diff | 0.015 (gstat: 0.013) | 0.04 |
| Ensemble std fields, mean abs diff / correlation | 0.020 / 0.889 | 0.04 / noise-aware* |
| Pooled mean / std / quantiles (q10–q90) | ≤0.005 | 0.02–0.03 |

\* The std field has low spatial contrast (spread 0.054 vs per-node MC
error ~0.016 at N = 1000), so the achievable engine-vs-engine correlation
is bounded by `spread² / (spread² + 2·se²)` ≈ 0.84 even for a perfect
simulator; the script derives this bound at runtime and requires 80% of it.

Both engines are equally close to the theoretical SK mean/std targets, so
the ensembles are statistically indistinguishable.

**Normal-score caveat documented by this exercise:** with `ties.method =
"average"` Walker's many V = 0 values collapse into a single transform knot
(score ≈ −1.85) and geostat-rs's range-clamped back-transform then truncates
the whole lower tail. The harness uses `ties.method = "first"` so the
internal transform reduces to the identity and the comparison isolates the
simulator. For real spiky data, despiking ahead of SGS is the user's
responsibility (as in GSLIB practice).

## v0.2 results (Meuse: KED, anisotropy, co-kriging)

Deterministic parity on meuse.grid (3103 cells), same models on both sides:

| Check | Predictions | Variances |
|---|---|---|
| KED `log(zinc) ~ sqrt(dist)` (gstat-fitted model) | 4.7e-13 | 1.4e-13 |
| OK with anisotropic model (Sph 900, anis 30°/0.5) | 2.0e-14 | 9.4e-16 |
| Ordinary co-kriging `log(zinc)+log(lead)`, `fit.lmc` LMC | 1.5e-12 | 1.2e-13 |

SIS has no direct gstat counterpart in this harness; it is covered by unit
tests (ensemble proportions track the global cdf, order-relation
corrections, reproducibility) plus a CLI smoke test on Walker Lake — the
auto-fitted indicator sills come out at the theoretical `p(1-p)`.

Reproduce v0.2:

```sh
Rscript validation/v02_gstat.R
BIN=target/release/geostat
$BIN krige -i validation/out/meuse_multi.csv --value-col lzinc \
    -m validation/out/gstat_ked_model.json \
    --drift-cols sdist --targets validation/out/grid_targets.csv \
    -o validation/out/rust_ked.csv
$BIN krige -i validation/out/meuse_multi.csv --value-col lzinc \
    -m validation/out/aniso_model.json \
    --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
    -o validation/out/rust_aniso.csv
$BIN cokrige -i validation/out/meuse_multi.csv --value-col lzinc \
    --secondary-col llead --lmc validation/out/gstat_lmc.json \
    --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
    -o validation/out/rust_cokrige.csv
python3 validation/compare_v02.py
```

## v0.3 results (Meuse: block kriging; Python bindings)

| Check | Predictions | Variances |
|---|---|---|
| Block kriging 40×40 m, explicit 4×4 discretization | 4.1e-14 | 1.5e-15 |
| Python bindings vs CLI (same model, 200 grid points) | 0.0 (bit-identical) | 0.0 |

Convention caught by this exercise: in `C̄(B,B)` gstat/GSLIB exclude the
nugget for coincident discretization points (a measure-zero discontinuity
in the block integral); including it shifts every block variance by
exactly `nugget / n_discr`. Reproduce: `Rscript validation/v03_gstat.R`,
then the `geostat krige --block` call in `compare_v03.py`'s docstring,
then `python3 validation/compare_v03.py`.

## v0.4 results (3-D, heterotopic co-kriging, indicator kriging)

| Check | Predictions | Variances |
|---|---|---|
| 3-D ordinary kriging (256 targets, fixed Sph model) | 9.5e-15 | 2.8e-15 |
| 3-D LOO cross-validation (200 points, RMSE) | matches to 1e-7 | — |
| Heterotopic co-kriging, shared LMC (meuse.grid, 3103 cells) | 2.6e-14 | 1.3e-15 |
| Indicator kriging, F(cutoff) where gstat stayed in [0,1] | 7.4e-10 | — |

The 3-D synthetic dataset and targets are shared between R and Rust
(`synth3d.csv`, `targets3d.csv`); the heterotopic case gives the secondary
variable (log lead) only half the locations of the primary (log zinc) and
injects the same LMC into both engines.

**Indicator-kriging note caught by this exercise:** with a single cutoff
gstat does plain simple kriging of the indicator and applies *no*
order-relation correction, so 500 of 3103 grid nodes get an estimated
probability outside `[0, 1]`. geostat-rs clamps `F` to a valid
probability; at those nodes the clamp is exactly `max(0, min(1, ·))` (0
mismatches), and everywhere gstat stayed in range the two engines agree to
7.4e-10. The clamp is the correct behavior — a ccdf value must be a
probability.

## v0.5 results (lognormal kriging, block co-kriging)

| Check | Predictions | Variances |
|---|---|---|
| Simple lognormal kriging vs gstat SK + analytic back-transform | 1.2e-9 | 7.3e-13 (log space) |
| Block co-kriging vs gstat `predict(block=)`, shared LMC (3103 cells) | 2.8e-14 | 1.4e-15 |

**Lognormal-kriging note caught by this exercise:** `gstat::krigeTg(lambda=0)`
does *not* implement the textbook ordinary lognormal back-transform. It
estimates the trend by GLS and adds a bias correction whose extra term is
~15× the ordinary-kriging Lagrange multiplier — a different (GLS-based)
trans-Gaussian estimator. So the clean oracle is **simple** lognormal
kriging, where the back-transform `exp(y + sigma2/2)` has no Lagrange term:
geostat-rs matches gstat's SK there to 1.2e-9. For ordinary lognormal
kriging geostat-rs uses the Journel & Huijbregts (1978) formula
`exp(y + sigma2/2 - mu)` (mu in covariance form); its log-space kriging is
gstat-validated to machine precision, but the OK back-transform is not
bit-compared to krigeTg.

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

Walker Lake + SGS:

```sh
Rscript validation/walker_gstat.R   # ~20 s (1000 sequential simulations)

$BIN variogram -i validation/out/walker_v.csv --value-col v \
    --n-lags 15 --max-dist 120 \
    -o validation/out/rust_walker_vario.csv \
    --fit spherical --model-out validation/out/rust_walker_model.json
$BIN krige -i validation/out/walker_v.csv --value-col v \
    -m validation/out/gstat_walker_model.json \
    --bbox 0,0,260,300 --nx 26 --ny 30 \
    -o validation/out/rust_walker_krige.csv
$BIN cv -i validation/out/walker_v.csv --value-col v \
    -m validation/out/gstat_walker_model.json \
    -o validation/out/rust_walker_cv.csv
$BIN sgs -i validation/out/walker_scores.csv --value-col score \
    --model-ns validation/out/gstat_ns_model.json \
    --bbox 0,0,260,300 --nx 26 --ny 30 \
    -n 1000 --seed 42 --max-neighbors 16 \
    -o validation/out/rust_sgs.csv

python3 validation/compare_walker.py
```

v0.4 (3-D, heterotopic co-kriging, indicator kriging):

```sh
Rscript validation/v04_gstat.R

$BIN krige -i validation/out/synth3d.csv --x-col x --y-col y --z-col z \
    --value-col v -m validation/out/model3d.json \
    --targets validation/out/targets3d.csv -o validation/out/rust_krige3d.csv
$BIN cv -i validation/out/synth3d.csv --x-col x --y-col y --z-col z \
    --value-col v -m validation/out/model3d.json   # prints RMSE
$BIN cokrige -i validation/out/meuse_primary.csv --value-col lzinc \
    --secondary-col llead --secondary-input validation/out/meuse_secondary.csv \
    --lmc validation/out/lmc_hetero.json \
    --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
    -o validation/out/rust_cokrige_hetero.csv
$BIN ik -i validation/out/meuse_ik.csv --value-col lzinc \
    --cutoffs "$(cat validation/out/ik_cutoff.txt)" \
    --models validation/out/ik_model.json \
    --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
    -o validation/out/rust_ik.csv

python3 validation/compare_v04.py
```

v0.5 (lognormal kriging, block co-kriging):

```sh
Rscript validation/v05_gstat.R

BETA=$(cat validation/out/logzinc_beta.txt)
$BIN krige -i validation/out/meuse_zinc.csv --value-col zinc \
    -m validation/out/logzinc_model.json --lognormal --method simple --mean "$BETA" \
    --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
    -o validation/out/rust_lognormal.csv
$BIN cokrige -i validation/out/meuse_multi2.csv --value-col lzinc \
    --secondary-col llead --lmc validation/out/lmc_block.json \
    --block 40,40 --block-discr 4,4 \
    --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
    -o validation/out/rust_block_cokrige.csv

python3 validation/compare_v05.py
```

`validation/out/` is regenerated on each run and not tracked by git.

## Notes

- The kriging/CV comparison uses **gstat's fitted model on both sides**, so
  it isolates the kriging engine from the fit. The fit comparison is
  reported separately (different optimizers; agreement to ~1e-6 relative
  on identical bins).
- Wall-clock for the 1000-realization SGS run: gstat ~18 s, geostat-rs
  ~2.4 s (same neighborhood settings, 8 threads).
