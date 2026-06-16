#!/usr/bin/env python3
"""Warped (transport) kriging vs plain ordinary kriging on heavily-skewed
rare-earth geochemistry — quantifying where Transport GPs earn their place.

Data: the national tailings geochemistry database from the TGPY project
(`tierras_raras.pkl`, 2032 samples). We take La(g/t) in the densest region
(Coquimbo, ~1100 deposits), which is extremely right-skewed (skew ~11,
five orders of magnitude). A random train/test split compares:

  * plain ordinary kriging on raw La, and
  * warped kriging with a fitted Box-Cox marginal,

on point accuracy AND, more importantly, on the predictive distribution:
80% interval coverage and physical validity (a concentration cannot be
negative — yet symmetric Gaussian kriging intervals on skewed data often
are). The warp's real value is calibrated, positive, asymmetric intervals.

Run: PYTHONPATH=<dir with geostat_rs.so> python3 examples/tgpy_rare_earths.py
"""

import math
import statistics as st
from pathlib import Path

import pandas as pd
import geostat_rs as gs

TGPY = Path.home() / "proyectos" / "TGPY" / "Python"
ELEMENT = "La(g/t)"
REGION = "COQUIMBO"
TEST_FRACTION = 0.25
SEED = 20260616


def load():
    d = pd.read_pickle(TGPY / "tierras_raras.pkl")

    def clean(col):
        s = d[col].astype(str).str.replace(",", ".", regex=False)
        s = s.str.replace("<", "", regex=False).str.replace(">", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    df = pd.DataFrame(
        {"x": clean("Coord. E"), "y": clean("Coord. N"),
         "v": clean(ELEMENT), "region": d["Region"]}
    )
    df = df[(df.x > 0) & (df.y > 0) & (df.v > 0)].dropna()
    df = df[df.region == REGION]
    # Many samples share a deposit location (identical coords) -> coincident
    # points make the kriging system singular. Average per location, as is
    # standard practice for collocated measurements.
    df = (
        df.assign(xr=df.x.round(0), yr=df.y.round(0))
        .groupby(["xr", "yr"], as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"), v=("v", "mean"))
    )
    return df.reset_index(drop=True)


def split(df):
    # Deterministic shuffle without numpy global state.
    idx = list(range(len(df)))
    rng = _LCG(SEED)
    for i in range(len(idx) - 1, 0, -1):
        j = rng.next() % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    n_test = int(len(df) * TEST_FRACTION)
    test = df.iloc[idx[:n_test]]
    train = df.iloc[idx[n_test:]]
    return train, test


class _LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s


def metrics(name, true, pred, lo, hi):
    n = len(true)
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, true)) / n)
    rmse_log = math.sqrt(
        sum((math.log(max(p, 1e-6)) - math.log(t)) ** 2 for p, t in zip(pred, true)) / n
    )
    coverage = sum(1 for t, a, b in zip(true, lo, hi) if a <= t <= b) / n
    width = st.median(b - a for a, b in zip(lo, hi))
    neg = sum(1 for a in lo if a < 0) / n
    print(
        f"  {name:<22} RMSE={rmse:7.1f}  RMSE(log)={rmse_log:5.3f}  "
        f"80%cover={coverage:4.0%}  width={width:7.1f}  neg.lower={neg:4.0%}"
    )
    return rmse, coverage


def main():
    df = load()
    print(f"geostat_rs {gs.__version__}")
    print(f"{ELEMENT} in {REGION}: {len(df)} samples")
    v = sorted(df.v)
    sk = (sum((a - st.mean(v)) ** 3 for a in v) / len(v)) / st.pstdev(v) ** 3
    print(f"  min={v[0]:.1f}  median={v[len(v)//2]:.1f}  max={v[-1]:.1f}  skew={sk:.1f}")

    train, test = split(df)
    tx, ty, tv = train.x.tolist(), train.y.tolist(), train.v.tolist()
    ex, ey, ev = test.x.tolist(), test.y.tolist(), test.v.tolist()
    print(f"  train={len(tx)}  test={len(ex)}  (max_neighbors=24)\n")

    # 1. Plain ordinary kriging on raw La. 80% Gaussian interval.
    model = gs.fit_variogram(tx, ty, tv, n_lags=12)
    pred, var = gs.krige(tx, ty, tv, model, ex, ey, method="ordinary", max_neighbors=24)
    z = 1.2815515  # 0.9 normal quantile
    ok_lo = [p - z * math.sqrt(max(s, 0)) for p, s in zip(pred, var)]
    ok_hi = [p + z * math.sqrt(max(s, 0)) for p, s in zip(pred, var)]

    # 2. Warped (Box-Cox) kriging. P10/P90 = 80% predictive interval.
    w = gs.warped_kriging(
        tx, ty, tv, ex, ey,
        warp="box-cox", quantiles=[0.1, 0.9],
        n_lags=12, max_neighbors=24, n_samples=4000, seed=1,
    )
    w_mean = w["mean"]
    w_lo = [q[0] for q in w["quantiles"]]
    w_hi = [q[1] for q in w["quantiles"]]

    print("Hold-out performance (80% predictive interval):")
    metrics("ordinary kriging", ev, pred, ok_lo, ok_hi)
    metrics("warped (Box-Cox)", ev, w_mean, w_lo, w_hi)
    print()
    print("Reading: coverage closer to 80% is better-calibrated; 'neg.lower'")
    print("is the fraction of lower bounds below 0 (impossible for a")
    print("concentration) — the warp keeps every interval physically valid.")


if __name__ == "__main__":
    main()
