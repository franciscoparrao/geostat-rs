#!/usr/bin/env python3
"""Numerical parity check: geostat-rs vs gstat (R) on the Meuse dataset.

Compares, for log(zinc):
  1. Experimental variogram bins (pair counts, mean distance, gamma).
  2. Fitted spherical model parameters (informative: different optimizers).
  3. Ordinary kriging predictions and variances on meuse.grid (3103 cells).
  4. Leave-one-out cross-validation predictions and variances (155 points).

Run after `Rscript validation/gstat_reference.R` and the geostat CLI calls
(see validation/README.md). Exits non-zero if any tolerance is violated.
"""

import csv
import math
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


def main():
    # --- 1. Experimental variogram ---------------------------------------
    print("1. Experimental variogram (15 lags, cutoff 1500)")
    gv = read_csv("gstat_vario.csv")
    rv = read_csv("rust_vario.csv")
    assert len(gv) == len(rv), f"bin count differs: {len(gv)} vs {len(rv)}"
    np_diff = max(abs(int(g["np"]) - int(r["n_pairs"])) for g, r in zip(gv, rv))
    h_diff = max(abs(float(g["dist"]) - float(r["h"])) for g, r in zip(gv, rv))
    g_diff = max(
        abs(float(g["gamma"]) - float(r["gamma"])) / float(g["gamma"])
        for g, r in zip(gv, rv)
    )
    check("pair counts, max abs diff", np_diff, 0)
    check("mean lag distance, max abs diff", h_diff, 1e-9)
    check("gamma, max rel diff", g_diff, 1e-12)

    # --- 2. Fitted model (informative) ------------------------------------
    print("2. Fitted spherical model (independent optimizers)")
    import json

    gm = {r["model"]: r for r in read_csv("gstat_model.csv")}
    rm = json.loads((OUT / "rust_model.json").read_text())
    nug_g = float(gm.get("Nug", {"psill": 0.0})["psill"])
    sill_g = float(gm["Sph"]["psill"])
    rng_g = float(gm["Sph"]["range"])
    check("nugget, rel diff", abs(rm["nugget"] - nug_g) / max(nug_g, 1e-12), 5e-2)
    check(
        "partial sill, rel diff",
        abs(rm["structures"][0]["sill"] - sill_g) / sill_g,
        5e-2,
    )
    check("range, rel diff", abs(rm["structures"][0]["range"] - rng_g) / rng_g, 5e-2)

    # --- 3. Ordinary kriging on meuse.grid --------------------------------
    print("3. Ordinary kriging, meuse.grid, global neighborhood, gstat model")
    rust = {
        (round(float(r["x"]), 3), round(float(r["y"]), 3)): r
        for r in read_csv("rust_krige.csv")
    }
    pred_diff = var_diff = 0.0
    n = 0
    for g in read_csv("gstat_krige.csv"):
        key = (round(float(g["x"]), 3), round(float(g["y"]), 3))
        r = rust.get(key)
        assert r is not None, f"grid cell {key} missing in rust output"
        pred_diff = max(pred_diff, abs(float(g["pred"]) - float(r["prediction"])))
        var_diff = max(var_diff, abs(float(g["var"]) - float(r["variance"])))
        n += 1
    print(f"  ({n} grid cells compared)")
    check("predictions, max abs diff", pred_diff, 1e-6)
    check("kriging variances, max abs diff", var_diff, 1e-6)

    # --- 4. Leave-one-out cross-validation --------------------------------
    print("4. Leave-one-out cross-validation (155 points)")
    rust_cv = {
        (round(float(r["x"]), 3), round(float(r["y"]), 3)): r
        for r in read_csv("rust_cv.csv")
    }
    pred_diff = var_diff = 0.0
    for g in read_csv("gstat_cv.csv"):
        key = (round(float(g["x"]), 3), round(float(g["y"]), 3))
        r = rust_cv.get(key)
        assert r is not None, f"CV point {key} missing in rust output"
        pred_diff = max(pred_diff, abs(float(g["pred"]) - float(r["predicted"])))
        var_diff = max(var_diff, abs(float(g["var"]) - float(r["variance"])))
    check("CV predictions, max abs diff", pred_diff, 1e-6)
    check("CV variances, max abs diff", var_diff, 1e-6)

    print()
    if FAILURES:
        print(f"PARITY FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("PARITY OK: geostat-rs matches gstat within tolerances.")


if __name__ == "__main__":
    main()
