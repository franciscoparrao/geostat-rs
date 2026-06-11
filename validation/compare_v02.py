#!/usr/bin/env python3
"""v0.2 parity check vs gstat on Meuse: KED, anisotropic OK, co-kriging.

All three comparisons are deterministic and use the same model on both
sides (gstat's fit for KED and the LMC; a fixed anisotropic model), so the
expected agreement is machine precision.

Run after `Rscript validation/v02_gstat.R` and the geostat CLI calls
(see validation/README.md). Exits non-zero on any violation.
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


def compare(title, gstat_file, rust_file, pred_field="prediction"):
    print(title)
    rust = {
        (round(float(r["x"]), 3), round(float(r["y"]), 3)): r for r in read_csv(rust_file)
    }
    pd = vd = 0.0
    n = 0
    for g in read_csv(gstat_file):
        key = (round(float(g["x"]), 3), round(float(g["y"]), 3))
        r = rust.get(key)
        assert r is not None, f"point {key} missing in {rust_file}"
        pd = max(pd, abs(float(g["pred"]) - float(r[pred_field])))
        vd = max(vd, abs(float(g["var"]) - float(r["variance"])))
        n += 1
    print(f"  ({n} locations compared)")
    check("predictions, max abs diff", pd, 1e-6)
    check("variances, max abs diff", vd, 1e-6)


def main():
    compare(
        "1. Kriging with external drift: log(zinc) ~ sqrt(dist), meuse.grid",
        "gstat_ked.csv",
        "rust_ked.csv",
    )
    compare(
        "2. Ordinary kriging, anisotropic model (Sph 900, anis 30deg/0.5)",
        "gstat_aniso.csv",
        "rust_aniso.csv",
    )
    compare(
        "3. Ordinary co-kriging log(zinc)+log(lead), fit.lmc LMC, global nbhd",
        "gstat_cokrige.csv",
        "rust_cokrige.csv",
    )

    print()
    if FAILURES:
        print(f"PARITY FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("V0.2 PARITY OK: KED, anisotropy and co-kriging match gstat.")


if __name__ == "__main__":
    main()
