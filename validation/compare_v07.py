#!/usr/bin/env python3
"""Numerical parity check v0.7: residual variograms vs gstat (Meuse).

gstat computes formula-based variograms on OLS residuals; this compares
  1. log(zinc) ~ sqrt(dist)  (external-drift residuals, --detrend-cols sdist)
  2. log(zinc) ~ x + y       (coordinate-trend residuals, --detrend 1)

Run after `Rscript validation/v07_gstat.R` and:
  BIN=target/release/geostat
  $BIN variogram -i validation/out/meuse_multi.csv --value-col lzinc \
      --detrend-cols sdist --n-lags 15 --max-dist 1500 \
      -o validation/out/rust_resid_drift_vario.csv
  $BIN variogram -i validation/out/meuse_multi.csv --value-col lzinc \
      --detrend 1 --n-lags 15 --max-dist 1500 \
      -o validation/out/rust_resid_poly_vario.csv

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
    if FAILURES:
        print(f"\nPARITY FAILED: {len(FAILURES)} check(s)")
        sys.exit(1)
    print("\nV0.7 PARITY OK: residual variograms match gstat.")


if __name__ == "__main__":
    main()
