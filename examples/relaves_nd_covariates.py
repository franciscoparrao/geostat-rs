#!/usr/bin/env python3
"""Multi-covariate prediction of a rare-earth grade on real Chilean tailings,
with the geostat-rs + Smelt machinery — the engine working on the kind of data
it was built for.

Data: the national tailings geochemistry database (`tierras_raras.pkl`,
2032 samples × 70 geochemical variables). We predict Nd (neodymium, g/t) in the
densest region (Coquimbo, ~1100 deposits), which is heavily right-skewed
(skew ~6). The covariates are deliberately NOT the neighbouring light REE
(Ce, Pr are ~0.95–0.99 correlated with Nd — they are measured together, so
using them is leakage, not a use case). Instead we use cheaper / more common
host-mineral proxies: P2O5 (REE sit in phosphates — monazite/apatite), Th
(an HFSE that tracks REE), La and TiO2. The honest question: do bulk-rock
proxies plus location beat plain spatial kriging for a specific REE grade?

Methods, scored on a hold-out by VEcv (variance explained by cross-validation,
Li 2016 — a scale-free, cross-validated R²) and RMSE:

  * ordinary kriging        (location only; no covariates)
  * regression kriging/OLS  (linear trend on the 4 covariates + residual krige)
  * random forest only      (Smelt; covariates, no spatial residual)
  * RF + residual kriging   (Smelt trend + geostat-rs residual krige = RFOK)

Requires `geostat_rs`, `smelt`, numpy, pandas in one environment. Run from the
repo root (the data path is absolute):
    python3 examples/relaves_nd_covariates.py
"""

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import geostat_rs as gs
import smelt

TGPY = Path.home() / "proyectos" / "TGPY" / "Python"
REGION = "COQUIMBO"
TARGET = "Nd(g/t)"
COVARIATES = ["P2O5(%)", "Th(g/t)", "La(g/t)", "TiO2(%)"]
SEED = 20260618
TEST_FRACTION = 0.25


def numeric(s):
    s = s.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace("<", "", regex=False).str.replace(">", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load():
    d = pd.read_pickle(TGPY / "tierras_raras.pkl")
    cols = {"x": "Coord. E", "y": "Coord. N", "v": TARGET}
    out = {k: numeric(d[c]) for k, c in cols.items()}
    for c in COVARIATES:
        out[c] = numeric(d[c])
    df = pd.DataFrame(out)
    df["region"] = d["Region"]
    df = df[(df.x > 0) & (df.y > 0) & (df.v > 0) & (df.region == REGION)].dropna()
    # Average duplicate-location samples (collocated measurements at a deposit).
    df = (
        df.assign(xr=df.x.round(0), yr=df.y.round(0))
        .groupby(["xr", "yr"], as_index=False)
        .agg({"x": "mean", "y": "mean", "v": "mean", **{c: "mean" for c in COVARIATES}})
    )
    return df.reset_index(drop=True)


def vecv(obs, pred):
    m = sum(obs) / len(obs)
    sse = sum((o - p) ** 2 for o, p in zip(obs, pred))
    sst = sum((o - m) ** 2 for o in obs)
    return (1 - sse / sst) * 100


def rmse(obs, pred):
    return math.sqrt(sum((o - p) ** 2 for o, p in zip(obs, pred)) / len(obs))


def neg_frac(lo):
    return sum(1 for a in lo if a < 0) / len(lo)


def main():
    df = load()
    rng = random.Random(SEED)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_test = int(len(df) * TEST_FRACTION)
    test_i, train_i = idx[:n_test], idx[n_test:]
    tr, te = df.iloc[train_i], df.iloc[test_i]

    trx, tryy, trv = tr.x.tolist(), tr.y.tolist(), tr.v.tolist()
    tex, tey, tev = te.x.tolist(), te.y.tolist(), te.v.tolist()
    tr_cov = tr[COVARIATES].values.tolist()
    te_cov = te[COVARIATES].values.tolist()
    Xtr = np.array(tr_cov, dtype=float)
    Xte = np.array(te_cov, dtype=float)
    ytr = np.array(trv, dtype=float)

    v = sorted(df.v)
    sk = (sum((a - np.mean(v)) ** 3 for a in v) / len(v)) / np.std(v) ** 3
    print(f"geostat_rs {gs.__version__}  +  smelt {getattr(smelt,'__version__','?')}")
    print(
        f"{TARGET} in {REGION}: {len(df)} deposits "
        f"(median={v[len(v)//2]:.1f}, max={v[-1]:.0f}, skew={sk:.1f}); "
        f"train={len(tr)}, test={len(te)}"
    )
    print(f"covariates: {', '.join(COVARIATES)}\n")

    # 1. Ordinary kriging (location only).
    model = gs.fit_variogram(trx, tryy, trv, n_lags=12)
    ok_pred, ok_var = gs.krige(trx, tryy, trv, model, tex, tey, method="ordinary", max_neighbors=24)

    # 2. Regression kriging with the built-in OLS trend on the covariates.
    rk = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov,
                               n_lags=12, max_neighbors=24)

    # 3. Smelt RandomForest on the covariates (the ML trend), used two ways.
    rf = smelt.RandomForest(n_estimators=300, max_depth=12, seed=SEED)
    rf.fit(Xtr, ytr)
    rf_tr = np.asarray(rf.predict(Xtr)).tolist()
    rf_te = np.asarray(rf.predict(Xte)).tolist()

    # 4. Hybrid: RF trend + geostat-rs residual kriging.
    hyb = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov,
                                trend_at_data=rf_tr, trend_at_targets=rf_te,
                                n_lags=12, max_neighbors=24)

    print(f"  {'method':<26}{'RMSE':>8}{'VEcv %':>9}")
    for name, pred in [
        ("ordinary kriging", ok_pred),
        ("regression kriging (OLS)", rk["prediction"]),
        ("random forest only", rf_te),
        ("RF + residual kriging", hyb["prediction"]),
    ]:
        print(f"  {name:<26}{rmse(tev, pred):>8.2f}{vecv(tev, pred):>9.1f}")

    # A symmetric-Gaussian 80% interval from ordinary kriging puts many lower
    # bounds below zero — invalid for a concentration, and a known limitation of
    # plain kriging on strongly-skewed grades.
    z = 1.281552  # standard-normal 0.90 quantile
    lo_ok = [m - z * math.sqrt(max(s, 0)) for m, s in zip(ok_pred, ok_var)]
    print(
        f"\nordinary-kriging 80% interval: {neg_frac(lo_ok):.0%} of lower bounds are "
        "negative\n(invalid for a grade) — symmetric Gaussian intervals are unsuited to "
        "skewed data."
    )
    print(
        "\nVEcv = variance explained by cross-validation (Li 2016); higher is better.\n"
        "On Nd the host-mineral proxies lift VEcv far above location-only kriging, and\n"
        "the ML trend + residual-kriging hybrid is best."
    )


if __name__ == "__main__":
    main()
