#!/usr/bin/env python3
"""Timing benchmark for geostat-rs (algorithm time only, via the Python
binding — a thin wrapper over the Rust compute). Companion to bench_gstat.R,
which times the identical operations in gstat. Both condition on the 155 Meuse
points and the same fitted spherical model, krige the same regular grids with a
moving neighbourhood (32 nearest), and run leave-one-out cross-validation on the
155 points. Data is preloaded; only the operation is timed (best of N_REP).

Writes the grids (shared with R) and bench/results_rust.csv.

Run from the repo root (after validation/gstat_reference.R):
    PYTHONPATH=<dir with geostat_rs.so> python3 bench/bench_rust.py
"""

import csv
import time
from pathlib import Path

import geostat_rs as gs

BENCH = Path("bench")
VAL = Path("validation/out")
GRID_SIZES = [50, 100, 200]  # n x n cells -> 2500, 10000, 40000
N_REP = 3


def load_meuse():
    with open(VAL / "meuse_lzinc.csv") as f:
        rows = list(csv.DictReader(f))
    x = [float(r["x"]) for r in rows]
    y = [float(r["y"]) for r in rows]
    v = [float(r["lzinc"]) for r in rows]
    return x, y, v


def make_grid(x, y, n):
    minx, maxx, miny, maxy = min(x), max(x), min(y), max(y)
    dx, dy = (maxx - minx) / n, (maxy - miny) / n
    tx, ty = [], []
    for j in range(n):
        for i in range(n):
            tx.append(minx + (i + 0.5) * dx)
            ty.append(miny + (j + 0.5) * dy)
    return tx, ty


def best(fn):
    t = float("inf")
    for _ in range(N_REP):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t


def main():
    BENCH.mkdir(exist_ok=True)
    x, y, v = load_meuse()
    model = gs.VariogramModel.from_json((VAL / "gstat_model.json").read_text())

    results = []
    for n in GRID_SIZES:
        tx, ty = make_grid(x, y, n)
        with open(BENCH / f"grid_{n}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y"])
            w.writerows(zip(tx, ty))
        t = best(lambda: gs.krige(x, y, v, model, tx, ty, method="ordinary", max_neighbors=32))
        results.append((f"OK grid {n*n}", t))
        print(f"  OK grid {n*n:>6} cells: {t*1000:8.1f} ms")

    t_cv = best(lambda: gs.loo_cv(x, y, v, model, max_neighbors=32))
    results.append(("LOO CV 155", t_cv))
    print(f"  LOO CV (155 points): {t_cv*1000:8.1f} ms")

    with open(BENCH / "results_rust.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seconds"])
        w.writerows(results)
    print("wrote bench/results_rust.csv")


if __name__ == "__main__":
    main()
