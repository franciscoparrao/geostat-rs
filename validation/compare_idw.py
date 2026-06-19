#!/usr/bin/env python3
"""Numerical parity check: geostat-rs IDW vs gstat IDW on the Meuse grid.

Reads the data (meuse_lzinc.csv) and gstat's IDW reference (gstat_idw.csv,
produced by idw_gstat.R), recomputes IDW with geostat-rs at the same grid
points (power 2, global neighbourhood) and reports the maximum difference.

Run from the repo root (after idw_gstat.R):
    PYTHONPATH=<dir with geostat_rs.so> python3 validation/compare_idw.py
"""

import csv
from pathlib import Path

import geostat_rs as gs

OUT = Path("validation/out")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    data = read_csv(OUT / "meuse_lzinc.csv")
    x = [float(r["x"]) for r in data]
    y = [float(r["y"]) for r in data]
    v = [float(r["lzinc"]) for r in data]

    ref = read_csv(OUT / "gstat_idw.csv")
    tx = [float(r["x"]) for r in ref]
    ty = [float(r["y"]) for r in ref]
    gstat_idw = [float(r["idw"]) for r in ref]

    rust_idw = gs.idw(x, y, v, tx, ty, power=2.0)  # global neighbourhood

    max_abs = max(abs(a - b) for a, b in zip(rust_idw, gstat_idw))
    denom = max(abs(b) for b in gstat_idw) or 1.0
    print(f"IDW vs gstat on {len(ref)} grid cells (power 2, global):")
    print(f"  max abs difference: {max_abs:.3e}")
    print(f"  max rel difference: {max_abs / denom:.3e}")
    print("  verdict:", "machine precision" if max_abs < 1e-9 else "CHECK")


if __name__ == "__main__":
    main()
