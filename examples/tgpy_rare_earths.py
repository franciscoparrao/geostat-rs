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
ELEMENTS = ["La(g/t)", "Ce(g/t)", "Nd(g/t)", "Y(g/t)"]
REGION = "COQUIMBO"
TEST_FRACTION = 0.25
SEED = 20260616


def load(element):
    d = pd.read_pickle(TGPY / "tierras_raras.pkl")

    def clean(col):
        s = d[col].astype(str).str.replace(",", ".", regex=False)
        s = s.str.replace("<", "", regex=False).str.replace(">", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    df = pd.DataFrame(
        {"x": clean("Coord. E"), "y": clean("Coord. N"),
         "v": clean(element), "region": d["Region"]}
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


# Standard-normal quantiles for the probabilities we evaluate.
ZQ = {
    0.05: -1.644854, 0.10: -1.281552, 0.15: -1.036433, 0.25: -0.674490,
    0.75: 0.674490, 0.85: 1.036433, 0.90: 1.281552, 0.95: 1.644854,
}
# Central predictive intervals to score, by nominal coverage.
LEVELS = {0.50: (0.25, 0.75), 0.70: (0.15, 0.85), 0.80: (0.10, 0.90), 0.90: (0.05, 0.95)}


def report(name, true, pred, quant):
    """quant: dict prob -> list of per-target quantile values."""
    n = len(true)
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, true)) / n)
    cov, cal_err = {}, 0.0
    for level, (lp, hp) in LEVELS.items():
        lo, hi = quant[lp], quant[hp]
        c = sum(1 for t, a, b in zip(true, lo, hi) if a <= t <= b) / n
        cov[level] = c
        cal_err += abs(c - level)
    cal_err /= len(LEVELS)
    neg = sum(1 for a in quant[0.10] if a < 0) / n  # lower bound of the 80% PI
    covs = "  ".join(f"{int(L*100)}%:{cov[L]:.0%}" for L in LEVELS)
    print(f"  {name:<20} RMSE={rmse:6.1f}  cover[{covs}]  cal.err={cal_err:.3f}  neg={neg:3.0%}")


def gaussian_quantiles(pred, var):
    """Quantile dict for a Gaussian predictive N(pred, var)."""
    std = [math.sqrt(max(s, 0)) for s in var]
    return {p: [m + ZQ[p] * s for m, s in zip(pred, std)] for p in ZQ}


def warp_quantiles(w):
    """Quantile dict from a warped_kriging result (probs aligned to PROBS)."""
    return {p: [q[i] for q in w["quantiles"]] for i, p in enumerate(PROBS)}


PROBS = [0.05, 0.10, 0.15, 0.25, 0.75, 0.85, 0.90, 0.95]


def run_element(element):
    df = load(element)
    v = sorted(df.v)
    sk = (sum((a - st.mean(v)) ** 3 for a in v) / len(v)) / st.pstdev(v) ** 3
    train, test = split(df)
    tx, ty, tv = train.x.tolist(), train.y.tolist(), train.v.tolist()
    ex, ey, ev = test.x.tolist(), test.y.tolist(), test.v.tolist()
    print(
        f"{element} in {REGION}: {len(df)} deposits  "
        f"(median={v[len(v)//2]:.1f}, max={v[-1]:.0f}, skew={sk:.1f}; "
        f"train={len(tx)}, test={len(ex)})"
    )

    model = gs.fit_variogram(tx, ty, tv, n_lags=12)
    pred, var = gs.krige(tx, ty, tv, model, ex, ey, method="ordinary", max_neighbors=24)

    def warp(kind, **kw):
        return gs.warped_kriging(
            tx, ty, tv, ex, ey,
            warp=kind, quantiles=PROBS,
            n_lags=12, max_neighbors=24, n_samples=8000, seed=1, **kw,
        )

    wb = warp("box-cox")
    ws = warp("sinh-arcsinh")
    # Auto: pick the marginal by AIC, clamped to non-negative concentrations.
    wa = warp("auto", floor=0.0)
    aic = "  ".join(f"{n}:{a:.0f}" for n, a in wa["aic_table"])
    report("ordinary kriging", ev, pred, gaussian_quantiles(pred, var))
    report("warped Box-Cox", ev, wb["mean"], warp_quantiles(wb))
    report("warped sinh-arcsinh", ev, ws["mean"], warp_quantiles(ws))
    report(f"warped auto[{wa['family']}]", ev, wa["mean"], warp_quantiles(wa))
    print(f"  AIC: {aic}")
    print()


def main():
    print(f"geostat_rs {gs.__version__}")
    print("Hold-out interval calibration vs nominal coverage; cal.err lower = better.\n")
    for element in ELEMENTS:
        run_element(element)
    print("cal.err = mean |empirical - nominal| over the four levels.")
    print("neg = fraction of 80%-interval lower bounds below 0 (invalid for a")
    print("concentration); the warps keep every interval positive.")


if __name__ == "__main__":
    main()
