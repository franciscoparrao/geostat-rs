#!/usr/bin/env python3
"""Numerical parity check: geostat-rs vs gstat (R) for the continuous-nu
Matern model (ModelKind::Matern), Meuse log(zinc), kappa/nu fixed at 1.2.

gstat's "Ste" model uses a different Matern range convention than this
engine's Rasmussen & Williams parameterization (range_rw = range_ste /
sqrt(2), independent of nu -- see validation/matern_gstat.R for the
derivation); this script applies that conversion before comparing.

Run after `Rscript validation/matern_gstat.R` and the geostat CLI call (see
validation/README.md). Exits non-zero if any tolerance is violated.
"""

import csv
import json
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
    print("Continuous-nu Matern (kappa/nu = 1.2), independent optimizers")
    gm = read_csv("gstat_matern_model.csv")[0]
    rm = json.loads((OUT / "rust_matern_model.json").read_text())

    nug_g = float(gm["nugget"])
    sill_g = float(gm["psill"])
    range_g_rw = float(gm["range_rw"])  # already converted to the R&W convention

    rust_kind = rm["structures"][0]["kind"]
    nu_rust = rust_kind["matern"] if isinstance(rust_kind, dict) else None
    if nu_rust is None or abs(nu_rust - float(gm["kappa"])) > 1e-9:
        print(f"  [FAIL] rust model kind is not matern:{gm['kappa']}: {rust_kind}")
        sys.exit(1)

    nug_r = rm["nugget"]
    sill_r = rm["structures"][0]["sill"]
    range_r = rm["structures"][0]["range"]

    check("nugget, rel diff", abs(nug_r - nug_g) / max(nug_g, 1e-12), 5e-2)
    check("partial sill, rel diff", abs(sill_r - sill_g) / max(sill_g, 1e-12), 5e-2)
    check(
        "range (R&W convention), rel diff",
        abs(range_r - range_g_rw) / max(range_g_rw, 1e-12),
        5e-2,
    )

    # Sanity check on the conversion itself: gstat's raw Ste range should be
    # range_rw * sqrt(2), independent of kappa.
    range_g_ste = float(gm["range_ste"])
    check(
        "range_ste / range_rw == sqrt(2), rel diff",
        abs(range_g_ste / range_g_rw - math.sqrt(2)) / math.sqrt(2),
        1e-9,
    )

    if FAILURES:
        print(f"\nPARITY FAILED: {len(FAILURES)} check(s) exceeded tolerance.")
        sys.exit(1)
    print("\nMATERN PARITY OK: geostat-rs matches gstat's Ste model within tolerances")
    print("(after converting range_ste -> range_rw = range_ste / sqrt(2)).")


if __name__ == "__main__":
    main()
