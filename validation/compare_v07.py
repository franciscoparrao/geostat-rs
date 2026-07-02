#!/usr/bin/env python3
"""Numerical parity check v0.7: residual variograms vs gstat (Meuse).

gstat computes formula-based variograms on OLS residuals; this compares
  1. log(zinc) ~ sqrt(dist)  (external-drift residuals, --detrend-cols sdist)
  2. log(zinc) ~ x + y       (coordinate-trend residuals, --detrend 1)

Also compares ordinary kriging with a constant measurement-error variance
(gstat's `Err` component vs `krige --error`).

Run after `Rscript validation/v07_gstat.R` and:
  BIN=target/release/geostat
  $BIN variogram -i validation/out/meuse_multi.csv --value-col lzinc \
      --detrend-cols sdist --n-lags 15 --max-dist 1500 \
      -o validation/out/rust_resid_drift_vario.csv
  $BIN variogram -i validation/out/meuse_multi.csv --value-col lzinc \
      --detrend 1 --n-lags 15 --max-dist 1500 \
      -o validation/out/rust_resid_poly_vario.csv
  $BIN krige -i validation/out/meuse_lzinc.csv --value-col lzinc \
      -m validation/out/gstat_model.json --error 0.05 \
      --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
      -o validation/out/rust_err_krige.csv

Exits non-zero if any tolerance is violated.
"""

import csv
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"
FAILURES = []


def read_csv(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def check(label, value, tol):
    status = "OK " if value <= tol else "FAIL"
    if value > tol:
        FAILURES.append(label)
    print(f"  [{status}] {label}: {value:.3e} (tol {tol:.0e})")


def compare(tag, gstat_name, rust_name):
    print(f"{tag}")
    gv = read_csv(gstat_name)
    rv = read_csv(rust_name)
    assert len(gv) == len(rv), f"bin count differs: {len(gv)} vs {len(rv)}"
    dn = max(abs(int(g["np"]) - int(float(r["n_pairs"]))) for g, r in zip(gv, rv))
    dh = max(abs(float(g["dist"]) - float(r["h"])) for g, r in zip(gv, rv))
    dg = max(
        abs(float(g["gamma"]) - float(r["gamma"])) / abs(float(g["gamma"]))
        for g, r in zip(gv, rv)
    )
    check("pair counts, max abs diff", dn, 0)
    check("mean lag distance, max abs diff", dh, 1e-9)
    check("gamma, max rel diff", dg, 1e-10)


def compare_err_kriging():
    print("3. Ordinary kriging with measurement error (gstat Err = 0.05)")
    rust = {
        (round(float(r["x"]), 3), round(float(r["y"]), 3)): r
        for r in read_csv("rust_err_krige.csv")
    }
    pred_diff = var_diff = 0.0
    for g in read_csv("gstat_err_krige.csv"):
        key = (round(float(g["x"]), 3), round(float(g["y"]), 3))
        r = rust.get(key)
        assert r is not None, f"grid cell {key} missing in rust output"
        pred_diff = max(pred_diff, abs(float(g["pred"]) - float(r["prediction"])))
        var_diff = max(var_diff, abs(float(g["var"]) - float(r["variance"])))
    check("predictions, max abs diff", pred_diff, 1e-6)
    check("kriging variances, max abs diff", var_diff, 1e-6)


def main():
    compare(
        "1. Residual variogram, external drift lzinc ~ sdist",
        "gstat_resid_drift_vario.csv",
        "rust_resid_drift_vario.csv",
    )
    compare(
        "2. Residual variogram, coordinate trend lzinc ~ x + y",
        "gstat_resid_poly_vario.csv",
        "rust_resid_poly_vario.csv",
    )
    compare_err_kriging()
    if FAILURES:
        print(f"\nPARITY FAILED: {len(FAILURES)} check(s)")
        sys.exit(1)
    print("\nV0.7 PARITY OK: residual variograms match gstat.")


if __name__ == "__main__":
    main()
