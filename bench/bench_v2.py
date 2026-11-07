#!/usr/bin/env python3
"""Timing benchmark v2: geostat-rs against compiled Python tools and gstat.

Companion: bench/bench_gstat_v2.R (gstat, same data/model/targets). Every tool
uses the SAME fixed variogram model, data and targets, so only the kriging step
is timed (data preloaded). Each timing is repeated (REPS) and all repetitions
are written, so the report can give median and spread.

Tasks
  meuse_grid_{2500,10000,40000}  OK, Meuse 155 points, 32 nearest neighbours
  synth_nmax_{1000,10000,100000} OK, n synthetic points, 10 000 targets, 32 nn
  synth_global_1000              OK, 1000 points, global neighbourhood, 10 000 targets
  sgs_synth_1000                 SGS, 1000 conditioning points, 50x50 grid,
                                 50 realizations, 16 nn (geostat-rs only here)

Tools: geostat-rs (threads from RAYON_NUM_THREADS), PyKrige 'C' backend
(moving neighbourhood, single-threaded), GSTools with the GSTools-Core Rust
backend (global neighbourhood only: it has no moving neighbourhood).

    python3 bench/bench_v2.py --write-data           # synthetic data + targets
    RAYON_NUM_THREADS=16 python3 bench/bench_v2.py --tools geostat-rs,pykrige,gstools
    RAYON_NUM_THREADS=1  python3 bench/bench_v2.py --tools geostat-rs --label 1
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

BENCH = Path("bench")
VAL = Path("validation/out")
OUT = BENCH / "results_v2.csv"
SYNTH_N = [1000, 10000, 100000]
# PyKrige's constructor evaluates the experimental variogram over all n^2/2
# pairs even when the model is given (87 s at n = 10^4 on the test machine),
# so n = 10^5 is out of reach; it is benchmarked up to this size.
PYKRIGE_MAX_N = 10000
EXTENT = 10_000.0
# Fixed model shared by every tool (nugget + spherical).
SYNTH_MODEL = {"nugget": 0.1, "psill": 0.9, "range": 2000.0}


def synth_points(n, seed):
    rng = random.Random(seed)
    x = [rng.uniform(0, EXTENT) for _ in range(n)]
    y = [rng.uniform(0, EXTENT) for _ in range(n)]
    v = [np.sin(a / 1500.0) + np.cos(b / 2100.0) + rng.gauss(0, 0.3) for a, b in zip(x, y)]
    return x, y, v


def targets_100x100():
    c = [(i + 0.5) * EXTENT / 100 for i in range(100)]
    return [cx for cy in c for cx in c], [cy for cy in c for cx in c]


def write_data():
    BENCH.mkdir(exist_ok=True)
    for n in SYNTH_N:
        x, y, v = synth_points(n, seed=n)
        with open(BENCH / f"synth_{n}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "v"])
            w.writerows(zip(x, y, v))
    tx, ty = targets_100x100()
    with open(BENCH / "synth_targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        w.writerows(zip(tx, ty))
    (BENCH / "synth_model.json").write_text(json.dumps(SYNTH_MODEL) + "\n")
    print("wrote synthetic data")


def read_xyv(path, cols=("x", "y", "v")):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [[float(r[c]) for r in rows] for c in cols]


def time_reps(fn, reps):
    out = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        out.append(time.perf_counter() - s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-data", action="store_true")
    ap.add_argument("--tools", default="geostat-rs")
    ap.add_argument("--label", default=os.environ.get("RAYON_NUM_THREADS", "all"))
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    if a.write_data:
        write_data()
        return
    tools = a.tools.split(",")
    rows, agree = [], {}

    def rec(tool, task, times):
        for i, t in enumerate(times):
            rows.append((tool, task, a.label, i, t))
        print(f"  {tool:11s} {task:22s} threads={a.label:>3}  median {np.median(times)*1e3:9.1f} ms", flush=True)

    # --- Meuse grids (model from the gstat parity work) ---
    mx, my, mv = read_xyv(VAL / "meuse_lzinc.csv", ("x", "y", "lzinc"))
    mjson = (VAL / "gstat_model.json").read_text()
    md = json.loads(mjson)
    st = md["structures"][0]
    for n in (50, 100, 200):
        tx, ty = read_xyv(BENCH / f"grid_{n}.csv", ("x", "y"))
        task = f"meuse_grid_{n*n}"
        reps = a.reps if n < 200 else max(3, a.reps // 2)
        if "geostat-rs" in tools:
            import geostat_rs as gs
            m = gs.VariogramModel.from_json(mjson)
            rec("geostat-rs", task, time_reps(lambda: gs.krige(mx, my, mv, m, tx, ty, method="ordinary", max_neighbors=32), reps))
            ref = np.asarray(gs.krige(mx, my, mv, m, tx, ty, method="ordinary", max_neighbors=32)[0])
        if "pykrige" in tools:
            from pykrige.ok import OrdinaryKriging
            ok = OrdinaryKriging(np.array(mx), np.array(my), np.array(mv), variogram_model="spherical",
                                 variogram_parameters={"psill": st["sill"], "range": st["range"], "nugget": md["nugget"]})
            f = lambda: ok.execute("points", np.array(tx), np.array(ty), backend="C", n_closest_points=32)
            rec("pykrige-C", task, time_reps(f, reps))
            if "geostat-rs" in tools:
                agree[f"pykrige_vs_geostat:{task}"] = float(np.max(np.abs(np.asarray(f()[0]) - ref)))

    # --- Synthetic scaling, moving neighbourhood ---
    sm = json.loads((BENCH / "synth_model.json").read_text())
    tx, ty = read_xyv(BENCH / "synth_targets.csv", ("x", "y"))
    sjson = json.dumps({"nugget": sm["nugget"], "structures": [
        {"kind": "spherical", "sill": sm["psill"], "range": sm["range"]}]})
    for n in SYNTH_N:
        x, y, v = read_xyv(BENCH / f"synth_{n}.csv")
        task = f"synth_nmax_{n}"
        if "geostat-rs" in tools:
            import geostat_rs as gs
            m = gs.VariogramModel.from_json(sjson)
            rec("geostat-rs", task, time_reps(lambda: gs.krige(x, y, v, m, tx, ty, method="ordinary", max_neighbors=32), a.reps))
            ref = np.asarray(gs.krige(x, y, v, m, tx, ty, method="ordinary", max_neighbors=32)[0])
        if "pykrige" in tools and n <= PYKRIGE_MAX_N:
            from pykrige.ok import OrdinaryKriging
            s0 = time.perf_counter()
            ok = OrdinaryKriging(np.array(x), np.array(y), np.array(v), variogram_model="spherical",
                                 variogram_parameters={"psill": sm["psill"], "range": sm["range"], "nugget": sm["nugget"]})
            agree[f"pykrige_constructor_seconds:{n}"] = time.perf_counter() - s0
            f = lambda: ok.execute("points", np.array(tx), np.array(ty), backend="C", n_closest_points=32)
            rec("pykrige-C", task, time_reps(f, a.reps))
            if "geostat-rs" in tools:
                agree[f"pykrige_vs_geostat:{task}"] = float(np.max(np.abs(np.asarray(f()[0]) - ref)))

    # --- Global neighbourhood, n = 1000 ---
    x, y, v = read_xyv(BENCH / "synth_1000.csv")
    task = "synth_global_1000"
    if "geostat-rs" in tools:
        import geostat_rs as gs
        m = gs.VariogramModel.from_json(sjson)
        rec("geostat-rs", task, time_reps(lambda: gs.krige(x, y, v, m, tx, ty, method="ordinary"), a.reps))
        ref = np.asarray(gs.krige(x, y, v, m, tx, ty, method="ordinary")[0])
    if "gstools" in tools:
        import gstools as gst
        gst.config.USE_GSTOOLS_CORE = True
        # GSTools Spherical: len_scale is the range; var is the partial sill.
        cm = gst.Spherical(dim=2, var=sm["psill"], len_scale=sm["range"], nugget=sm["nugget"])
        krig = gst.krige.Ordinary(cm, cond_pos=[np.array(x), np.array(y)], cond_val=np.array(v), exact=True)
        f = lambda: krig([np.array(tx), np.array(ty)], return_var=False)
        rec("gstools-rs", task, time_reps(f, a.reps))
        if "geostat-rs" in tools:
            agree[f"gstools_vs_geostat:{task}"] = float(np.max(np.abs(np.asarray(f()) - ref)))

    # --- SGS (geostat-rs; gstat in the R companion) ---
    if "geostat-rs" in tools:
        import geostat_rs as gs
        m = gs.VariogramModel.from_json(sjson)
        x, y, v = read_xyv(BENCH / "synth_1000.csv")
        rec("geostat-rs", "sgs_synth_1000", time_reps(
            lambda: gs.sgs(x, y, v, m, (0, 0, EXTENT, EXTENT), 50, 50, n_realizations=50, seed=1, max_neighbors=16),
            max(3, a.reps // 2)))

    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tool", "task", "threads", "rep", "seconds"])
        w.writerows(rows)
    if agree:
        p = BENCH / "agreement_v2.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(agree)
        p.write_text(json.dumps(old, indent=1) + "\n")
        print("max |difference| between tools:", {k: f"{v:.2e}" for k, v in agree.items()})


if __name__ == "__main__":
    main()
