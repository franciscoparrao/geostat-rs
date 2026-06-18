#!/usr/bin/env python3
"""Ordinary co-kriging of a rare earth on real tailings — borrowing strength
from a densely-measured correlated REE when the target is undersampled.

Co-kriging earns its keep in the *heterotopic* case: the variable you care
about is expensive / sparsely assayed, while a correlated one is measured
everywhere. We emulate that on the Coquimbo tailings (TGPY tierras_raras):
Nd (neodymium) is the primary and La (lanthanum) the secondary — both light
REE, moderately coregionalized in log space (r ≈ 0.62). A moderate correlation
is deliberate: a near-perfectly-correlated dense secondary (e.g. Ce/Pr/Gd at
r > 0.9) drives the fitted LMC to the positive-definite boundary and the full
co-kriging system turns near-singular, so the standard estimator becomes
unstable — La keeps it well-conditioned while still carrying real signal.

Setup: a 25% hold-out is kept to score Nd predictions. In the training set the
secondary (La) is kept at EVERY location, but the primary (Nd) is thinned to a
fraction (the "expensive assays"); the rest are La-only sites. We compare, by
VEcv on the held-out Nd:

  * ordinary kriging of Nd, sparse primary only
  * ordinary co-kriging of Nd with the dense La secondary (auto-fitted LMC)
  * ordinary kriging of Nd, full primary (reference upper bound)

across decreasing primary-sampling fractions. The sparser the primary, the more
co-kriging should recover by leaning on the dense secondary.

Requires geostat_rs + numpy + pandas. Run from the repo root:
    python3 examples/relaves_cokriging.py
"""

import math
import random
from pathlib import Path

import pandas as pd

import geostat_rs as gs

TGPY = Path.home() / "proyectos" / "TGPY" / "Python"
REGION = "COQUIMBO"
PRIMARY = "Nd(g/t)"
SECONDARY = "La(g/t)"
SEED = 20260618
TEST_FRACTION = 0.25
PRIMARY_FRACTIONS = [0.5, 0.3, 0.15]


def numeric(s):
    s = s.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace("<", "", regex=False).str.replace(">", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load():
    d = pd.read_pickle(TGPY / "tierras_raras.pkl")
    df = pd.DataFrame(
        {
            "x": numeric(d["Coord. E"]),
            "y": numeric(d["Coord. N"]),
            "p": numeric(d[PRIMARY]),
            "s": numeric(d[SECONDARY]),
        }
    )
    df["region"] = d["Region"]
    df = df[(df.x > 0) & (df.y > 0) & (df.p > 0) & (df.s > 0) & (df.region == REGION)]
    df = df.drop(columns="region").dropna()
    df = (
        df.assign(xr=df.x.round(0), yr=df.y.round(0))
        .groupby(["xr", "yr"], as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"), p=("p", "mean"), s=("s", "mean"))
    )
    # Work in log space: REE grades are strongly right-skewed (skew ~6), where
    # raw-value variography and kriging are numerically unstable, especially
    # with a thinned primary. Logs are the standard, well-behaved setting and
    # keep the OK-vs-co-kriging comparison fair (both predict log-grade).
    df["p"] = df["p"].map(math.log)
    df["s"] = df["s"].map(math.log)
    return df.reset_index(drop=True)


def vecv(obs, pred):
    m = sum(obs) / len(obs)
    sse = sum((o - q) ** 2 for o, q in zip(obs, pred))
    sst = sum((o - m) ** 2 for o in obs)
    return (1 - sse / sst) * 100


def main():
    df = load()
    rng = random.Random(SEED)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_test = int(len(df) * TEST_FRACTION)
    test, train = df.iloc[idx[:n_test]], df.iloc[idx[n_test:]]

    ex, ey, ev = test.x.tolist(), test.y.tolist(), test.p.tolist()
    # The full training primary and the dense secondary.
    full_px, full_py, full_pv = train.x.tolist(), train.y.tolist(), train.p.tolist()
    sx, sy, sv = train.x.tolist(), train.y.tolist(), train.s.tolist()

    print(f"geostat_rs {gs.__version__}")
    print(
        f"{REGION}: predict log {PRIMARY} (primary) from log {SECONDARY} (secondary), "
        f"corr≈{train.p.corr(train.s):.2f}"
    )
    print(f"deposits={len(df)}  train={len(train)}  test={len(test)}\n")

    # Reference: ordinary kriging with the FULL primary (Nd dense everywhere).
    full_model = gs.fit_variogram(full_px, full_py, full_pv, n_lags=12)
    full_pred, _ = gs.krige(full_px, full_py, full_pv, full_model, ex, ey,
                            method="ordinary", max_neighbors=16)
    full_vecv = vecv(ev, full_pred)

    # A single random thinning is noisy, so average each fraction over several
    # independent thinnings (the secondary stays dense throughout).
    n_rep = 8
    print(f"  {'primary frac':<14}{'~n primary':>11}{'OK VEcv':>10}{'coK VEcv':>10}{'gain':>8}"
          f"   (mean over {n_rep} thinnings)")
    rng2 = random.Random(SEED + 1)
    for frac in PRIMARY_FRACTIONS:
        ok_vs, ck_vs, ns = [], [], []
        for _ in range(n_rep):
            keep = [i for i in range(len(train)) if rng2.random() < frac]
            px = [full_px[i] for i in keep]
            py = [full_py[i] for i in keep]
            pv = [full_pv[i] for i in keep]
            ok_model = gs.fit_variogram(px, py, pv, n_lags=12)
            ok_pred, _ = gs.krige(px, py, pv, ok_model, ex, ey, method="ordinary", max_neighbors=16)
            ck_pred, _ = gs.co_kriging(px, py, pv, sx, sy, sv, ex, ey,
                                       n_lags=12, max_neighbors=16, ridge=1e-2)
            ok_vs.append(vecv(ev, ok_pred))
            ck_vs.append(vecv(ev, ck_pred))
            ns.append(len(keep))
        ok_v = sum(ok_vs) / n_rep
        ck_v = sum(ck_vs) / n_rep
        print(
            f"  {frac:<14.2f}{sum(ns) // n_rep:>11}{ok_v:>10.1f}{ck_v:>10.1f}{ck_v - ok_v:>+8.1f}"
        )

    print(f"\n  full primary (reference, {len(train)} pts):  OK VEcv={full_vecv:.1f}")
    print(
        "\nVEcv = variance explained by cross-validation (Li 2016). With the dense La\n"
        "secondary, co-kriging gives a small gain over ordinary kriging at moderate\n"
        "undersampling (clearest around a third of sites); at the sparsest the noisy\n"
        "cross-variogram erodes it. The gain is modest because this regional REE field\n"
        "has weak spatial structure (full-primary VEcv ~23) and La is only moderately\n"
        "correlated with Nd. Notes on numerical stability: the\n"
        "co-kriging system is ill-conditioned here, so a ridge (1e-2) is needed; a\n"
        "near-perfectly-correlated dense secondary (Ce/Pr/Gd, r>0.9) drives the LMC to\n"
        "the PSD boundary and destabilizes it further — moderate correlation is safer."
    )


if __name__ == "__main__":
    main()
