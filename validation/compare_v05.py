#!/usr/bin/env python3
"""v0.5 parity check vs gstat: lognormal kriging and block co-kriging.

- Simple lognormal kriging: the back-transform exp(y + sigma2/2) has no
  Lagrange term, so gstat SK + the analytic back-transform is an
  unambiguous oracle. Both the log-space variance and the back-transformed
  prediction are matched to machine precision. (The ordinary case uses the
  Journel & Huijbregts formula; its log-space kriging is gstat-validated,
  but gstat::krigeTg uses a different GLS-based correction so it is not a
  bit oracle for OK — see validation/README.md.)
- Block co-kriging vs gstat predict(..., block=) on a cokriging object,
  shared LMC and 4x4 discretization of 40 m blocks.

Run after `Rscript validation/v05_gstat.R` and the geostat CLI calls in
validation/README.md.
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


def key2(r):
    return (round(float(r["x"]), 3), round(float(r["y"]), 3))


def main():
    # ---- 1. Lognormal kriging ----
    print("1. Simple lognormal kriging vs gstat SK + analytic back-transform")
    rust = {key2(r): r for r in read_csv("rust_lognormal.csv")}
    log_pred_d = log_var_d = pred_d = 0.0
    n = 0
    for g in read_csv("gstat_lognormal.csv"):
        r = rust.get(key2(g))
        if r is None:
            continue
        # Rust grid CSV stores back-transformed value in 'prediction' and the
        # log-space variance in 'variance'.
        pred_d = max(pred_d, abs(float(g["pred"]) - float(r["prediction"])))
        log_var_d = max(log_var_d, abs(float(g["log_var"]) - float(r["variance"])))
        n += 1
    print(f"  ({n} matched cells)")
    check("log-space variance, max abs diff", log_var_d, 1e-6)
    check("back-transformed prediction, max abs diff", pred_d, 1e-5)

    # ---- 2. Block co-kriging ----
    print("2. Block co-kriging vs gstat predict(block=) (shared LMC)")
    rust = {key2(r): r for r in read_csv("rust_block_cokrige.csv")}
    pd = vd = 0.0
    n = 0
    for g in read_csv("gstat_block_cokrige.csv"):
        r = rust.get(key2(g))
        if r is None:
            continue
        pd = max(pd, abs(float(g["pred"]) - float(r["prediction"])))
        vd = max(vd, abs(float(g["var"]) - float(r["variance"])))
        n += 1
    print(f"  ({n} matched cells)")
    check("predictions, max abs diff", pd, 1e-6)
    check("block variances, max abs diff", vd, 1e-6)

    print()
    if FAILURES:
        print(f"PARITY FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("V0.5 PARITY OK: lognormal kriging and block co-kriging match gstat.")


if __name__ == "__main__":
    main()
