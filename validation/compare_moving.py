#!/usr/bin/env python3
"""Parity of geostat-rs with gstat for moving-neighbourhood kriging (Meuse).

Run after `Rscript validation/moving_gstat.R`, with the Python module on the
path. Compares predictions and variances per case (OK with nmax 8/16/32, OK
with a 600 m search radius, SK with nmax 16, UK with a linear drift and nmax
32) and leave-one-out CV with nmax 32. Exits non-zero on any failure.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import geostat_rs as gs

OUT = Path(__file__).parent / "out"
TOL = 1e-6
CASES = {
    "ok_nmax8": dict(method="ordinary", max_neighbors=8),
    "ok_nmax16": dict(method="ordinary", max_neighbors=16),
    "ok_nmax32": dict(method="ordinary", max_neighbors=32),
    "ok_nmax32_radius600": dict(method="ordinary", max_neighbors=32, radius=600.0),
    "sk_nmax16": dict(method="simple", max_neighbors=16),
    "uk1_nmax32": dict(method="universal", degree=1, max_neighbors=32),
}


def rows(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def main():
    d = rows("meuse_lzinc.csv")
    x, y, v = ([float(r[c]) for r in d] for c in ("x", "y", "lzinc"))
    model = gs.VariogramModel.from_json((OUT / "gstat_model.json").read_text())
    by_case = defaultdict(list)
    for r in rows("gstat_moving.csv"):
        by_case[r["case"]].append(r)
    fails = []
    print(f"Moving-neighbourhood parity vs gstat (Meuse log-zinc, {len(by_case['ok_nmax8'])} cells)")
    for case, kw in CASES.items():
        g = by_case[case]
        tx, ty = [float(r["x"]) for r in g], [float(r["y"]) for r in g]
        gp = np.array([float(r["pred"]) if r["pred"] != "NA" else np.nan for r in g])
        gv = np.array([float(r["var"]) if r["var"] != "NA" else np.nan for r in g])
        if kw["method"] == "simple":
            kw = dict(kw, mean=float(np.mean(v)))
        p, var = gs.krige(x, y, v, model, tx, ty, **kw)
        p, var = np.asarray(p), np.asarray(var)
        both = ~np.isnan(gp) & np.isfinite(p)
        dp, dv = np.max(np.abs(p[both] - gp[both])), np.max(np.abs(var[both] - gv[both]))
        na = int((~both).sum())
        ok = dp <= TOL and dv <= TOL
        fails += [] if ok else [case]
        print(f"  [{'OK ' if ok else 'FAIL'}] {case:22s} pred {dp:.2e}  var {dv:.2e}"
              + (f"  ({na} cells without neighbours on one side)" if na else ""))
    cvg = rows("gstat_moving_cv.csv")
    cv = gs.loo_cv(x, y, v, model, method="ordinary", max_neighbors=32)
    dp = np.max(np.abs(np.asarray(cv["predicted"]) - np.array([float(r["pred"]) for r in cvg])))
    dv = np.max(np.abs(np.asarray(cv["variance"]) - np.array([float(r["var"]) for r in cvg])))
    ok = dp <= TOL and dv <= TOL
    fails += [] if ok else ["loo_nmax32"]
    print(f"  [{'OK ' if ok else 'FAIL'}] {'loo_nmax32':22s} pred {dp:.2e}  var {dv:.2e}")
    if fails:
        print("FAILED:", ", ".join(fails))
        sys.exit(1)
    print("MOVING-NEIGHBOURHOOD PARITY OK")


if __name__ == "__main__":
    main()
