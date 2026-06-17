#!/usr/bin/env python3
"""Regression kriging vs ordinary kriging on the Meuse data — the value of a
trend covariate, scored by VEcv (variance explained by cross-validation).

Regression kriging (RK) fits a trend in a first step (here OLS of log-zinc on
`sdist`, the standardized distance to the river Meuse) and then kriges the
residuals, adding the trend back at the targets. Because the trend is fitted
*separately*, it can come from any regressor — OLS here, or a machine-learning
model (e.g. Smelt) for a full ML+geostatistics hybrid.

On a random hold-out we compare plain ordinary kriging (no covariate) against
regression kriging on `sdist`, using RMSE and VEcv (Li 2016): a cross-validated,
scale-free R² where 100 = perfect and 0 = no better than the mean. The river
distance is informative, so RK should win.

Run:  PYTHONPATH=<dir with geostat_rs.so> python3 examples/regression_kriging_meuse.py
      (run from the repo root so validation/out/meuse_multi.csv resolves)
"""

import csv
import math
import random
from pathlib import Path

import geostat_rs as gs

SEED = 20260617
TEST_FRACTION = 0.25
DATA = Path("validation/out/meuse_multi.csv")  # x, y, lzinc, llead, sdist


def load():
    rows = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            rows.append(
                (float(r["x"]), float(r["y"]), float(r["lzinc"]), float(r["sdist"]))
            )
    return rows


def vecv(obs, pred):
    m = sum(obs) / len(obs)
    sse = sum((o - p) ** 2 for o, p in zip(obs, pred))
    sst = sum((o - m) ** 2 for o in obs)
    return (1 - sse / sst) * 100


def rmse(obs, pred):
    return math.sqrt(sum((o - p) ** 2 for o, p in zip(obs, pred)) / len(obs))


def main():
    rows = load()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_test = int(len(rows) * TEST_FRACTION)
    test, train = rows[:n_test], rows[n_test:]

    tx = [r[0] for r in train]; ty = [r[1] for r in train]
    tv = [r[2] for r in train]; tc = [[r[3]] for r in train]
    ex = [r[0] for r in test]; ey = [r[1] for r in test]
    ev = [r[2] for r in test]; ec = [[r[3]] for r in test]

    # Ordinary kriging (no covariate).
    model = gs.fit_variogram(tx, ty, tv, n_lags=15)
    ok_pred, _ = gs.krige(tx, ty, tv, model, ex, ey, method="ordinary", max_neighbors=32)

    # Regression kriging on sdist (OLS trend + residual kriging).
    rk = gs.regression_kriging(tx, ty, tv, tc, ex, ey, ec, n_lags=15, max_neighbors=32)
    rk_pred = rk["prediction"]

    b = rk["trend_coef"]
    print(f"geostat_rs {gs.__version__}")
    print(
        f"Meuse log-zinc, hold-out (train={len(train)}, test={len(test)}), covariate = sdist"
    )
    print(f"  OLS trend: lzinc = {b[0]:.4f} {b[1]:+.4f}*sdist")
    print(f"  {'method':<22}{'RMSE':>8}{'VEcv %':>10}")
    print(f"  {'ordinary kriging':<22}{rmse(ev, ok_pred):>8.4f}{vecv(ev, ok_pred):>10.2f}")
    print(f"  {'regression kriging':<22}{rmse(ev, rk_pred):>8.4f}{vecv(ev, rk_pred):>10.2f}")
    print("\nVEcv = variance explained by cross-validation (Li 2016); higher is better.")


if __name__ == "__main__":
    main()
