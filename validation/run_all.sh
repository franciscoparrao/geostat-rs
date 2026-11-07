#!/usr/bin/env bash
# Regenerates EVERY geostat-rs output used by the parity comparators from the
# current code, then runs all comparators. The gstat references in
# validation/out/gstat_* are left as they are (they do not depend on this
# code); regenerate them with the Rscript lines in README.md if needed.
#
# Stale rust_* outputs are moved aside first, so a comparator whose input was
# not regenerated fails instead of silently re-checking an old result.
#
#   bash validation/run_all.sh        # needs the Python module on the path
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p geostat-cli
BIN=target/release/geostat
O=validation/out
STALE="$O/stale_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STALE"
mv "$O"/rust_* "$STALE"/ 2>/dev/null || true
G="--bbox 178440,329600,181560,333760 --nx 78 --ny 104"

# v0.1 Meuse
$BIN variogram -i $O/meuse_lzinc.csv --value-col lzinc --n-lags 15 --max-dist 1500 -o $O/rust_vario.csv --fit spherical --model-out $O/rust_model.json
$BIN krige -i $O/meuse_lzinc.csv --value-col lzinc -m $O/gstat_model.json $G -o $O/rust_krige.csv
$BIN cv -i $O/meuse_lzinc.csv --value-col lzinc -m $O/gstat_model.json -o $O/rust_cv.csv
# v0.2
$BIN krige -i $O/meuse_multi.csv --value-col lzinc -m $O/gstat_ked_model.json --drift-cols sdist --targets $O/grid_targets.csv -o $O/rust_ked.csv
$BIN krige -i $O/meuse_multi.csv --value-col lzinc -m $O/aniso_model.json $G -o $O/rust_aniso.csv
$BIN cokrige -i $O/meuse_multi.csv --value-col lzinc --secondary-col llead --lmc $O/gstat_lmc.json $G -o $O/rust_cokrige.csv
# v0.3 block
$BIN krige -i $O/meuse_lzinc.csv --value-col lzinc -m $O/gstat_model.json --block 40,40 --block-discr 4,4 $G -o $O/rust_block.csv
# Matérn
$BIN variogram -i $O/meuse_lzinc.csv --value-col lzinc --n-lags 15 --max-dist 1500 -o $O/rust_matern_vario.csv --fit "matern:1.2" --model-out $O/rust_matern_model.json
# Walker Lake
$BIN variogram -i $O/walker_v.csv --value-col v --n-lags 15 --max-dist 120 -o $O/rust_walker_vario.csv --fit spherical --model-out $O/rust_walker_model.json
$BIN krige -i $O/walker_v.csv --value-col v -m $O/gstat_walker_model.json --bbox 0,0,260,300 --nx 26 --ny 30 -o $O/rust_walker_krige.csv
$BIN cv -i $O/walker_v.csv --value-col v -m $O/gstat_walker_model.json -o $O/rust_walker_cv.csv
$BIN sgs -i $O/walker_scores.csv --value-col score --model-ns $O/gstat_ns_model.json --bbox 0,0,260,300 --nx 26 --ny 30 -n 1000 --seed 42 --max-neighbors 16 -o $O/rust_sgs.csv
# v0.4
$BIN krige -i $O/synth3d.csv --x-col x --y-col y --z-col z --value-col v -m $O/model3d.json --targets $O/targets3d.csv -o $O/rust_krige3d.csv
$BIN cokrige -i $O/meuse_primary.csv --value-col lzinc --secondary-col llead --secondary-input $O/meuse_secondary.csv --lmc $O/lmc_hetero.json $G -o $O/rust_cokrige_hetero.csv
$BIN ik -i $O/meuse_ik.csv --value-col lzinc --cutoffs "$(cat $O/ik_cutoff.txt)" --models $O/ik_model.json $G -o $O/rust_ik.csv
# v0.5
BETA=$(cat $O/logzinc_beta.txt)
$BIN krige -i $O/meuse_zinc.csv --value-col zinc -m $O/logzinc_model.json --lognormal --method simple --mean "$BETA" $G -o $O/rust_lognormal.csv
$BIN cokrige -i $O/meuse_multi2.csv --value-col lzinc --secondary-col llead --lmc $O/lmc_block.json --block 40,40 --block-discr 4,4 $G -o $O/rust_block_cokrige.csv
# v0.7
$BIN variogram -i $O/meuse_multi.csv --value-col lzinc --detrend-cols sdist --n-lags 15 --max-dist 1500 -o $O/rust_resid_drift_vario.csv
$BIN variogram -i $O/meuse_multi.csv --value-col lzinc --detrend 1 --n-lags 15 --max-dist 1500 -o $O/rust_resid_poly_vario.csv
$BIN krige -i $O/meuse_lzinc.csv --value-col lzinc -m $O/gstat_model.json --error 0.05 $G -o $O/rust_err_krige.csv

fail=0
for c in compare compare_v02 compare_v03 compare_v04 compare_v05 compare_v07 compare_idw compare_matern compare_walker compare_moving; do
  echo "=== $c"
  if python3 validation/$c.py > /tmp/geostat_parity_$c.log 2>&1; then tail -1 /tmp/geostat_parity_$c.log; else echo "FAILED (see /tmp/geostat_parity_$c.log)"; tail -15 /tmp/geostat_parity_$c.log; fail=1; fi
done
n_stale=$(ls "$STALE" | wc -l)
echo "regenerated $(ls $O/rust_* | wc -l) outputs; $n_stale stale files moved to $STALE"
exit $fail
