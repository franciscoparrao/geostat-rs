#!/usr/bin/env python3
"""Multi-element rare-earth grade prediction on real tailings — extending the
single-element study (`relaves_nd_covariates.py`) across the REE series.

For each target REE we predict its grade in Coquimbo from the SAME fixed set of
non-REE host-mineral proxies — P2O5, Th (phosphates / monazite, which carry the
light REE), and TiO2, Fe2O3, Zr (Ti–Fe oxides and zircon, which carry the heavy
REE and Y). None of the covariates is a rare earth, so there is no REE-on-REE
leakage for any target; the random forest is free to pick whichever proxies a
given element actually tracks. This mirrors the real geochemistry: light REE
partition into phosphates, heavy REE and Y into zircon / heavy minerals.

Each element is scored on a hold-out by VEcv (variance explained by
cross-validation, Li 2016) for four predictors. The spread of skew across the
series (Y ≈ 0.5 to La ≈ 10) shows where the covariates earn their place — and
where ordinary kriging's symmetric intervals fail on skewed grades.

Requires geostat_rs + smelt + numpy + pandas. Run from the repo root:
    python3 examples/relaves_multielement.py
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
TARGETS = ["La(g/t)", "Ce(g/t)", "Nd(g/t)", "Dy(g/t)", "Y(g/t)"]
COVARIATES = ["P2O5(%)", "Th(g/t)", "TiO2(%)", "Fe2O3(%)", "Zr(g/t)"]
SEED = 20260618
TEST_FRACTION = 0.25
Z90 = 1.281552  # standard-normal 0.90 quantile (for an 80% interval)


def numeric(s):
    s = s.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace("<", "", regex=False).str.replace(">", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load():
    d = pd.read_pickle(TGPY / "tierras_raras.pkl")
    cols = {"x": "Coord. E", "y": "Coord. N"}
    out = {k: numeric(d[c]) for k, c in cols.items()}
    for c in TARGETS + COVARIATES:
        out[c] = numeric(d[c])
    df = pd.DataFrame(out)
    df["region"] = d["Region"]
    df = df[(df.x > 0) & (df.y > 0) & (df.region == REGION)].drop(columns="region").dropna()
    # Average collocated samples (same deposit location).
    agg = {"x": "mean", "y": "mean", **{c: "mean" for c in TARGETS + COVARIATES}}
    df = df.assign(xr=df.x.round(0), yr=df.y.round(0)).groupby(["xr", "yr"], as_index=False).agg(agg)
    return df.reset_index(drop=True)


def vecv(obs, pred):
    m = sum(obs) / len(obs)
    sse = sum((o - p) ** 2 for o, p in zip(obs, pred))
    sst = sum((o - m) ** 2 for o in obs)
    return (1 - sse / sst) * 100


def skew(v):
    a = np.asarray(v)
    return float(((a - a.mean()) ** 3).mean() / a.std() ** 3)


def main():
    df = load()
    rng = random.Random(SEED)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_test = int(len(df) * TEST_FRACTION)
    tr, te = df.iloc[idx[n_test:]], df.iloc[idx[:n_test]]

    trx, tryy = tr.x.tolist(), tr.y.tolist()
    tex, tey = te.x.tolist(), te.y.tolist()
    tr_cov, te_cov = tr[COVARIATES].values.tolist(), te[COVARIATES].values.tolist()
    Xtr, Xte = np.array(tr_cov, float), np.array(te_cov, float)

    print(f"geostat_rs {gs.__version__}  +  smelt {getattr(smelt,'__version__','?')}")
    print(f"{REGION}: {len(df)} deposits  (train={len(tr)}, test={len(te)})")
    print(f"covariates (non-REE host proxies): {', '.join(COVARIATES)}\n")
    print("Predictive accuracy — VEcv % (higher is better):")
    print(f"  {'element':<9}{'skew':>6}{'OK':>8}{'RK-OLS':>8}{'RF':>8}{'hybrid':>8}")

    neg_rows = []
    for tgt in TARGETS:
        trv = tr[tgt].tolist()
        tev = te[tgt].tolist()

        model = gs.fit_variogram(trx, tryy, trv, n_lags=12)
        ok_pred, ok_var = gs.krige(trx, tryy, trv, model, tex, tey,
                                   method="ordinary", max_neighbors=24)
        rk = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov,
                                   n_lags=12, max_neighbors=24)
        rf = smelt.RandomForest(n_estimators=300, max_depth=12, seed=SEED)
        rf.fit(Xtr, np.array(trv, float))
        rf_tr, rf_te = np.asarray(rf.predict(Xtr)).tolist(), np.asarray(rf.predict(Xte)).tolist()
        hyb = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov,
                                    trend_at_data=rf_tr, trend_at_targets=rf_te,
                                    n_lags=12, max_neighbors=24)

        name = tgt.split("(")[0]
        print(f"  {name:<9}{skew(df[tgt].tolist()):>6.1f}{vecv(tev, ok_pred):>8.1f}"
              f"{vecv(tev, rk['prediction']):>8.1f}{vecv(tev, rf_te):>8.1f}"
              f"{vecv(tev, hyb['prediction']):>8.1f}")

        # Fraction of ordinary-kriging 80% lower bounds below 0 (invalid grade).
        lo_ok = [m - Z90 * math.sqrt(max(s, 0)) for m, s in zip(ok_pred, ok_var)]
        neg_rows.append((name, sum(1 for a in lo_ok if a < 0) / len(lo_ok)))

    print("\nOrdinary-kriging 80% interval: fraction of lower bounds below 0")
    print("(invalid for a grade) — symmetric Gaussian intervals fail on skew:")
    print(f"  {'element':<9}{'neg lower bounds':>18}")
    for name, neg in neg_rows:
        print(f"  {name:<9}{neg:>17.0%}")

    print("\nLight REE (Ce, Nd) track the phosphate proxies (P2O5, Th); Y (heavy)")
    print("tracks Ti/Fe/Zr; La is poorly predicted by any of them. Covariates help")
    print("in proportion to that signal. The negative-bound fraction grows with skew.")


if __name__ == "__main__":
    main()
