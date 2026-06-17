#!/usr/bin/env python3
"""ML + geostatistics hybrid, fully Rust-native: a Smelt machine-learning trend
plus residual kriging in geostat-rs — the RFOK / "random forest + ordinary
kriging" method of Jin Li (Spatial Predictive Modeling with R, 2021), built
from the author's own Rust engines instead of R/scikit-learn.

The idea (Hengl et al.; Li 2021): a flexible regressor captures the trend
m(x) = f(covariates), and kriging models the spatially-correlated residuals
r(x) = z(x) - m(x). The prediction is z*(x0) = m*(x0) + r*(x0). Li notes
(§12.3) that ML models alone give poor spatial uncertainty — kriging the
residuals restores a geostatistical predictive variance.

geostat-rs's `regression_kriging` takes the trend evaluated at the data and at
the targets, so ANY regressor can supply it. Here Smelt (smelt-ml) provides a
RandomForest trend; geostat-rs kriges the residuals. We compare four predictors
on a Meuse log-zinc hold-out, scored by VEcv (variance explained by
cross-validation, Li 2016 — a scale-free, cross-validated R²):

  * ordinary kriging        (no covariates; pure spatial)
  * random forest only      (Smelt trend, no residual kriging)
  * regression kriging/OLS  (geostat-rs built-in linear trend + residuals)
  * RF + residual kriging   (Smelt RF trend + geostat-rs residuals = RFOK)

The trend uses the one covariate available everywhere, sdist (distance to the
river) — NOT the coordinates. That is deliberate: if the RF is given x/y it
does its own spatial interpolation and the residual kriging adds little
("RFsp", Hengl 2018). Feeding it only the covariate isolates the hybrid's
point — the RF learns the covariate response, kriging supplies the spatial
structure the RF cannot see — which is the classic RFOK setting and the
realistic prediction scenario (covariate rasters exist everywhere; the target
values do not).

Requires `smelt` and `geostat_rs` in the same environment, plus numpy. Run
from the repo root so validation/out/meuse_multi.csv resolves:
    python3 examples/hybrid_smelt_rk.py
"""

import csv
import math
import random
from pathlib import Path

import numpy as np

import geostat_rs as gs
import smelt

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

    # Coordinates and target.
    trx = [r[0] for r in train]; tryy = [r[1] for r in train]; trv = [r[2] for r in train]
    tex = [r[0] for r in test]; tey = [r[1] for r in test]; tev = [r[2] for r in test]

    # Trend feature matrix: the covariate sdist only (no coordinates — see the
    # module docstring). Same columns for the OLS rows and the RF.
    def feats(rows_):
        return [[r[3]] for r in rows_]

    train_feats = feats(train)
    test_feats = feats(test)
    Xtr = np.array(train_feats, dtype=float)
    Xte = np.array(test_feats, dtype=float)
    ytr = np.array(trv, dtype=float)

    print(f"geostat_rs {gs.__version__}  +  smelt {getattr(smelt, '__version__', '?')}")
    print(
        f"Meuse log-zinc hold-out (train={len(train)}, test={len(test)}); "
        f"trend covariate = sdist (no coordinates)\n"
    )

    # 1. Ordinary kriging — no covariates.
    model = gs.fit_variogram(trx, tryy, trv, n_lags=15)
    ok_pred, _ = gs.krige(trx, tryy, trv, model, tex, tey,
                          method="ordinary", max_neighbors=32)

    # 2. Smelt RandomForest trend (the ML mean), used two ways below.
    rf = smelt.RandomForest(n_estimators=300, max_depth=12, seed=SEED)
    rf.fit(Xtr, ytr)
    rf_train = np.asarray(rf.predict(Xtr)).tolist()  # trend at data
    rf_test = np.asarray(rf.predict(Xte)).tolist()   # trend at targets

    # 3. Regression kriging with the built-in OLS trend on the same features.
    rk_ols = gs.regression_kriging(
        trx, tryy, trv, train_feats, tex, tey, test_feats,
        n_lags=15, max_neighbors=32,
    )

    # 4. Hybrid: Smelt RF trend + geostat-rs residual kriging (RFOK).
    #    Pass the RF trend directly via trend_at_data / trend_at_targets.
    rk_rf = gs.regression_kriging(
        trx, tryy, trv, train_feats, tex, tey, test_feats,
        trend_at_data=rf_train, trend_at_targets=rf_test,
        n_lags=15, max_neighbors=32,
    )

    print(f"  {'method':<26}{'RMSE':>8}{'VEcv %':>10}")
    rf_vecv = vecv(tev, rf_test)
    hybrid_vecv = vecv(tev, rk_rf["prediction"])
    rowsout = [
        ("ordinary kriging", ok_pred),
        ("random forest only", rf_test),
        ("regression kriging (OLS)", rk_ols["prediction"]),
        ("RF + residual kriging", rk_rf["prediction"]),
    ]
    for name, pred in rowsout:
        print(f"  {name:<26}{rmse(tev, pred):>8.4f}{vecv(tev, pred):>10.2f}")

    print(
        "\nVEcv = variance explained by cross-validation (Li 2016); higher is better."
    )
    print(
        f"Residual kriging lifts the RF trend by {hybrid_vecv - rf_vecv:+.1f} VEcv points "
        f"({rf_vecv:.1f} -> {hybrid_vecv:.1f}):\nit recovers the spatial structure the "
        "covariate-only RF cannot see (and adds a geostatistical\npredictive variance the "
        "RF lacks). On Meuse the log-zinc/sdist relationship is\nnearly linear, so the OLS "
        "trend wins outright here — a reminder that no method\ndominates and methods must be "
        "compared by predictive accuracy (Li 2021)."
    )


if __name__ == "__main__":
    main()
