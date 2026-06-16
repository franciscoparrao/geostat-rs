#!/usr/bin/env python3
"""Warped (transport) kriging on real mine-tailings geochemistry — the
bridge between geostat-rs and the TGPY project in action.

Conditioning data: 15 ground measurements of Cu(g/t) at the Dulcinea
tailings (small, skewed sample — the regime where Transport GPs shine).
We warped-krige Cu over the same prediction grid the original `tgpy`
pipeline used, and compare:

  * the fitted marginal warp (Box-Cox lambda),
  * warped E-type vs plain ordinary kriging,
  * the predictive P20/P80 spread vs the original tgpy output.

This is a sanity demonstration, not a parity test: the original tgpy run
regressed Cu on spectral bands, whereas this is a spatial warped kriging
on (x, y). The point is that geostat-rs's transport kriging produces
sensible, uncertainty-aware predictions on the real tiny sample.

Run:  PYTHONPATH=<dir with geostat_rs.so> python3 examples/tgpy_relaves_dulcinea.py
"""

import csv
import math
import statistics as st
from pathlib import Path

import geostat_rs as gs

TGPY = Path.home() / "proyectos" / "TGPY" / "Python"


def load_measurements():
    x, y, cu = [], [], []
    with open(TGPY / "modelo_1_cu_v1.csv") as f:
        for r in csv.DictReader(f):
            x.append(float(r["x"]))
            y.append(float(r["y"]))
            cu.append(float(r["Cu(g/t)"]))
    return x, y, cu


def load_tgpy_grid():
    tx, ty, mean, q80, q20 = [], [], [], [], []
    with open(TGPY / "resultados_tgp.csv") as f:
        for r in csv.DictReader(f):
            tx.append(float(r["Coord. E"]))
            ty.append(float(r["Coord. N"]))
            mean.append(float(r["Mean"]))
            q80.append(float(r["Quantile 80"]))
            q20.append(float(r["Quantile 20"]))
    return tx, ty, mean, q80, q20


def summary(name, v):
    v = [a for a in v if math.isfinite(a)]
    v.sort()
    print(
        f"  {name:<26} n={len(v):4d}  min={v[0]:8.0f}  med={v[len(v)//2]:8.0f}"
        f"  mean={st.mean(v):8.0f}  max={v[-1]:8.0f}"
    )


def main():
    x, y, cu = load_measurements()
    tx, ty, t_mean, t_q80, t_q20 = load_tgpy_grid()
    print(f"geostat_rs {gs.__version__}")
    print(f"Conditioning: {len(cu)} Cu(g/t) measurements at Dulcinea")
    summary("measured Cu(g/t)", cu)
    print(f"Prediction grid: {len(tx)} nodes (from resultados_tgp.csv)\n")

    # 1. Warped (transport) kriging, Box-Cox marginal, with P20/P50/P80.
    warped = gs.warped_kriging(
        x, y, cu, tx, ty,
        warp="box-cox",
        quantiles=[0.2, 0.5, 0.8],
        n_lags=10,
        n_samples=4000,
        seed=7,
    )
    w_mean = warped["mean"]
    w_q20 = [q[0] for q in warped["quantiles"]]
    w_q80 = [q[2] for q in warped["quantiles"]]

    # 2. Plain ordinary kriging for comparison (fit a model in data space).
    ok_model = gs.fit_variogram(x, y, cu, n_lags=10)
    ok_pred, _ = gs.krige(x, y, cu, ok_model, tx, ty, method="ordinary")

    print("Predicted Cu(g/t) over the grid:")
    summary("geostat-rs warped E-type", w_mean)
    summary("geostat-rs ordinary K", ok_pred)
    summary("original tgpy Mean", t_mean)
    print()
    print("Predictive uncertainty (P80 - P20 spread), median over grid:")
    w_spread = st.median(b - a for a, b in zip(w_q20, w_q80))
    t_spread = st.median(b - a for a, b in zip(t_q20, t_q80))
    print(f"  geostat-rs warped : {w_spread:8.0f} g/t")
    print(f"  original tgpy     : {t_spread:8.0f} g/t")
    print()

    # 3. How much does the warp matter? Correlation warped-vs-OK and the
    #    fraction where the warped mean exceeds the OK mean (skew correction).
    finite = [
        (w, o)
        for w, o in zip(w_mean, ok_pred)
        if math.isfinite(w) and math.isfinite(o)
    ]
    n = len(finite)
    mw = sum(w for w, _ in finite) / n
    mo = sum(o for _, o in finite) / n
    cov = sum((w - mw) * (o - mo) for w, o in finite)
    vw = sum((w - mw) ** 2 for w, _ in finite)
    vo = sum((o - mo) ** 2 for _, o in finite)
    corr = cov / math.sqrt(vw * vo)
    print(f"warped vs ordinary kriging: corr={corr:.3f}, "
          f"mean(warped)-mean(OK)={mw - mo:+.0f} g/t")

    # 4. Write a CSV the TGPY notebooks can pick up.
    out = Path(__file__).parent / "dulcinea_warped_kriging.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["x", "y", "cu_mean", "cu_p20", "cu_p80", "cu_ok"])
        for i in range(len(tx)):
            wr.writerow([tx[i], ty[i], w_mean[i], w_q20[i], w_q80[i], ok_pred[i]])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
