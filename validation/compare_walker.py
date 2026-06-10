#!/usr/bin/env python3
"""Walker Lake parity + distributional SGS check: geostat-rs vs gstat.

Part A (deterministic parity, V variable, 470 points):
  experimental variogram, fitted model, OK on a 26x30 grid, LOO CV.

Part B (distributional, normal scores of V):
  1000 conditional Gaussian simulations per engine (SK mean 0, 16 neighbors)
  with different RNGs, so only ensemble statistics are comparable:
    - per-node ensemble mean field vs gstat's, and both vs the SK prediction
      (the ensemble mean must converge to it as N grows);
    - per-node ensemble std field, with a noise-aware correlation threshold:
      the std field has low spatial contrast, so the achievable
      engine-vs-engine correlation is limited by Monte Carlo error even for
      a perfect simulator — the bound is derived from the theoretical SK std
      field and se(s) = s/sqrt(2(N-1));
    - pooled moments and quantiles over all draws.

Run after walker_gstat.R and the geostat CLI calls (validation/README.md).
Exits non-zero if any tolerance is violated.
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


def check(label, value, tol, mode="max"):
    ok = value <= tol if mode == "max" else value >= tol
    status = "OK " if ok else "FAIL"
    if not ok:
        FAILURES.append(label)
    rel = "min" if mode == "min" else "tol"
    print(f"  [{status}] {label}: {value:.4g} ({rel} {tol:g})")


def key(row, xf="x", yf="y"):
    return (round(float(row[xf]), 3), round(float(row[yf]), 3))


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb)


def main():
    # ---------- Part A: deterministic parity ------------------------------
    print("A1. Experimental variogram, V (15 lags, cutoff 120)")
    gv = read_csv("gstat_walker_vario.csv")
    rv = read_csv("rust_walker_vario.csv")
    assert len(gv) == len(rv)
    check(
        "pair counts, max abs diff",
        max(abs(int(g["np"]) - int(r["n_pairs"])) for g, r in zip(gv, rv)),
        0,
    )
    check(
        "mean lag distance, max abs diff",
        max(abs(float(g["dist"]) - float(r["h"])) for g, r in zip(gv, rv)),
        1e-9,
    )
    check(
        "gamma, max rel diff",
        max(
            abs(float(g["gamma"]) - float(r["gamma"])) / float(g["gamma"])
            for g, r in zip(gv, rv)
        ),
        1e-12,
    )

    print("A2. Fitted spherical model, V (independent optimizers)")
    import json

    gm = {r["model"]: r for r in read_csv("gstat_walker_model.csv")}
    rm = json.loads((OUT / "rust_walker_model.json").read_text())
    nug_g = float(gm.get("Nug", {"psill": 0.0})["psill"])
    check("nugget, rel diff", abs(rm["nugget"] - nug_g) / max(nug_g, 1e-12), 5e-2)
    check(
        "partial sill, rel diff",
        abs(rm["structures"][0]["sill"] - float(gm["Sph"]["psill"]))
        / float(gm["Sph"]["psill"]),
        5e-2,
    )
    check(
        "range, rel diff",
        abs(rm["structures"][0]["range"] - float(gm["Sph"]["range"]))
        / float(gm["Sph"]["range"]),
        5e-2,
    )

    print("A3. Ordinary kriging, 780 cells, global neighborhood, gstat model")
    rust = {key(r): r for r in read_csv("rust_walker_krige.csv")}
    pd = vd = 0.0
    for g in read_csv("gstat_walker_krige.csv"):
        r = rust[key(g)]
        pd = max(pd, abs(float(g["pred"]) - float(r["prediction"])))
        vd = max(vd, abs(float(g["var"]) - float(r["variance"])))
    check("predictions, max abs diff", pd, 1e-6)
    check("kriging variances, max abs diff", vd, 1e-6)

    print("A4. Leave-one-out cross-validation (470 points)")
    rust_cv = {key(r): r for r in read_csv("rust_walker_cv.csv")}
    pd = vd = 0.0
    for g in read_csv("gstat_walker_cv.csv"):
        r = rust_cv[key(g)]
        pd = max(pd, abs(float(g["pred"]) - float(r["predicted"])))
        vd = max(vd, abs(float(g["var"]) - float(r["variance"])))
    check("CV predictions, max abs diff", pd, 1e-6)
    check("CV variances, max abs diff", vd, 1e-6)

    # ---------- Part B: distributional SGS --------------------------------
    gn = read_csv("gstat_sgs_nodes.csv")
    rust_rows = {key(r): r for r in read_csv("rust_sgs.csv")}
    nsim = sum(1 for c in read_csv("rust_sgs.csv")[0] if c.startswith("sim"))
    print(f"B. SGS, normal scores: {nsim} realizations/engine, SK(0), nmax 16")
    assert nsim == 1000, f"expected 1000 realizations, got {nsim}"

    g_mean, g_std, r_mean, r_std, sk_pred = [], [], [], [], []
    pooled_r = []
    for g in gn:
        r = rust_rows[key(g)]
        sims = [float(r[f"sim{i}"]) for i in range(1, nsim + 1)]
        m = sum(sims) / nsim
        s = math.sqrt(sum((x - m) ** 2 for x in sims) / (nsim - 1))
        r_mean.append(m)
        r_std.append(s)
        pooled_r.extend(sims)
        g_mean.append(float(g["mean"]))
        g_std.append(float(g["std"]))
        sk_pred.append(float(g["sk_pred"]))

    n_nodes = len(g_mean)
    print(f"  ({n_nodes} nodes compared)")

    rmse_mean = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(r_mean, g_mean)) / n_nodes
    )
    check("ensemble mean fields, RMSE", rmse_mean, 0.08)
    check("ensemble mean fields, correlation", corr(r_mean, g_mean), 0.98, "min")

    # Both ensemble means must sit on the SK prediction (within MC error).
    rmse_r_sk = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(r_mean, sk_pred)) / n_nodes
    )
    rmse_g_sk = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(g_mean, sk_pred)) / n_nodes
    )
    print(f"  (rust mean vs SK: {rmse_r_sk:.4f}; gstat mean vs SK: {rmse_g_sk:.4f})")
    check("rust ensemble mean vs SK prediction, RMSE", rmse_r_sk, 0.06)

    # Each engine's ensemble std against the theoretical SK std (the marginal
    # posterior std, up to the nmax approximation) — the primary correctness
    # check for the spread.
    sk_std = [math.sqrt(float(g["sk_var"])) for g in gn]
    d_r = sum(abs(a - b) for a, b in zip(r_std, sk_std)) / n_nodes
    d_g = sum(abs(a - b) for a, b in zip(g_std, sk_std)) / n_nodes
    print(f"  (std vs theoretical SK std, mean abs diff — gstat: {d_g:.4f})")
    check("rust ensemble std vs theoretical SK std, mean abs diff", d_r, 0.04)

    mean_std_diff = sum(abs(a - b) for a, b in zip(r_std, g_std)) / n_nodes
    check("ensemble std fields, mean abs diff", mean_std_diff, 0.04)

    # Noise-aware correlation bound: with per-node MC error se(s) and a std
    # field of limited spatial contrast, even a perfect simulator cannot
    # exceed corr_max = spread^2 / (spread^2 + 2 se^2). Require 80% of it.
    ms = sum(sk_std) / n_nodes
    spread2 = sum((s - ms) ** 2 for s in sk_std) / n_nodes
    se2 = (ms / math.sqrt(2.0 * (nsim - 1))) ** 2
    corr_bound = spread2 / (spread2 + 2.0 * se2)
    print(f"  (theoretical max std-field correlation at N={nsim}: {corr_bound:.3f})")
    check(
        "ensemble std fields, correlation",
        corr(r_std, g_std),
        0.8 * corr_bound,
        "min",
    )

    gp = read_csv("gstat_sgs_pooled.csv")[0]
    np_ = len(pooled_r)
    pm = sum(pooled_r) / np_
    ps = math.sqrt(sum((x - pm) ** 2 for x in pooled_r) / (np_ - 1))
    check("pooled mean, abs diff", abs(pm - float(gp["mean"])), 0.02)
    check("pooled std, rel diff", abs(ps - float(gp["std"])) / float(gp["std"]), 0.02)
    pooled_r.sort()
    for q, col in [(0.10, "q10"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.90, "q90")]:
        rq = pooled_r[int(q * np_)]
        check(f"pooled q{int(q*100)}, abs diff", abs(rq - float(gp[col])), 0.03)

    print()
    if FAILURES:
        print(f"PARITY FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("WALKER LAKE OK: parity at machine precision; SGS ensembles "
          "statistically indistinguishable.")


if __name__ == "__main__":
    main()
