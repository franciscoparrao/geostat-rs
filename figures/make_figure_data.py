#!/usr/bin/env python3
"""Generates the data tables for the paper figures (figures/figures.R reads
them). Publishable methods only — no transport/warped kriging.

Outputs to figures/data/:
  parity.csv        geostat-rs vs gstat (OK and IDW) on the Meuse grid
  compare_vecv.csv  leave-one-out VEcv by method on Meuse log-zinc
  idw_tune.csv      IDW power tuning trace (VEcv vs power) on Meuse log-zinc
  multielement.csv  REE hold-out VEcv by element x method (needs smelt + the
                    tailings dataset; skipped if unavailable)

Run from the repo root, after `Rscript validation/gstat_reference.R` and
`Rscript validation/idw_gstat.R`:
    PYTHONPATH=<dir with geostat_rs.so> python3 figures/make_figure_data.py
"""

import csv
import math
import random
from pathlib import Path

import geostat_rs as gs

OUT = Path("figures/data")
VAL = Path("validation/out")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def vecv(obs, pred):
    m = sum(obs) / len(obs)
    sse = sum((o - p) ** 2 for o, p in zip(obs, pred))
    sst = sum((o - m) ** 2 for o in obs)
    return (1 - sse / sst) * 100


def meuse():
    d = read_csv(VAL / "meuse_lzinc.csv")
    return (
        [float(r["x"]) for r in d],
        [float(r["y"]) for r in d],
        [float(r["lzinc"]) for r in d],
    )


def fig_parity():
    """geostat-rs vs gstat for OK and IDW on the Meuse grid (machine-precision
    agreement; the validation centrepiece)."""
    x, y, v = meuse()
    model = gs.VariogramModel.from_json((VAL / "gstat_model.json").read_text())

    ok = read_csv(VAL / "gstat_krige.csv")
    tx = [float(r["x"]) for r in ok]
    ty = [float(r["y"]) for r in ok]
    rust_ok, _ = gs.krige(x, y, v, model, tx, ty, method="ordinary")  # global nbhd
    rows = [("ordinary kriging", float(r["pred"]), p) for r, p in zip(ok, rust_ok)]

    idw = read_csv(VAL / "gstat_idw.csv")
    rust_idw = gs.idw(x, y, v, [float(r["x"]) for r in idw], [float(r["y"]) for r in idw], power=2.0)
    rows += [("IDW (power 2)", float(r["idw"]), p) for r, p in zip(idw, rust_idw)]

    write_csv(OUT / "parity.csv", ["method", "gstat", "rust"], rows)


def fig_compare():
    """Leave-one-out VEcv by method on Meuse log-zinc."""
    x, y, v = meuse()
    cmp = gs.compare_methods(x, y, v, max_neighbors=32, knn_k=8)
    label = {
        "ordinary_kriging": "ordinary kriging",
        "idw": "IDW",
        "knn": "k-NN",
        "nearest_neighbor": "nearest neighbour",
    }
    rows = [(label[k], m["vecv"], m["rmse"]) for k, m in cmp.items()]
    rows.sort(key=lambda r: -r[1])
    write_csv(OUT / "compare_vecv.csv", ["method", "vecv", "rmse"], rows)


def fig_idw_tune():
    """IDW power tuning trace (VEcv vs power)."""
    x, y, v = meuse()
    res = gs.tune_idw_power(x, y, v, powers=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])
    rows = [(p, s) for p, s in res["trace"]]
    write_csv(OUT / "idw_tune.csv", ["power", "vecv"], rows)
    print(f"  best IDW power = {res['best']} (VEcv {res['best_vecv']:.1f})")


def fig_multielement():
    """REE hold-out VEcv by element x method (publishable methods only).

    Data provenance: the national tailings geochemistry database; confirm it is
    clear to publish before including this figure. No transport kriging here.
    """
    try:
        import numpy as np
        import pandas as pd
        import smelt
    except ImportError as e:
        print(f"  multielement skipped (missing {e.name})")
        return
    data = Path.home() / "proyectos" / "TGPY" / "Python" / "tierras_raras.pkl"
    if not data.exists():
        print("  multielement skipped (dataset not found)")
        return

    region = "COQUIMBO"
    targets = ["La(g/t)", "Ce(g/t)", "Nd(g/t)", "Dy(g/t)", "Y(g/t)"]
    covs = ["P2O5(%)", "Th(g/t)", "TiO2(%)", "Fe2O3(%)", "Zr(g/t)"]
    seed = 20260618

    def num(s):
        s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s.str.replace("<", "", regex=False).str.replace(">", "", regex=False),
                             errors="coerce")

    d = pd.read_pickle(data)
    df = pd.DataFrame({"x": num(d["Coord. E"]), "y": num(d["Coord. N"])})
    for c in targets + covs:
        df[c] = num(d[c])
    df["r"] = d["Region"]
    df = df[(df.x > 0) & (df.y > 0) & (df.r == region)].drop(columns="r").dropna()
    agg = {"x": "mean", "y": "mean", **{c: "mean" for c in targets + covs}}
    df = df.assign(xr=df.x.round(0), yr=df.y.round(0)).groupby(["xr", "yr"], as_index=False).agg(agg)

    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    nt = len(df) // 4
    tr, te = df.iloc[idx[nt:]], df.iloc[idx[:nt]]
    trx, tryy = tr.x.tolist(), tr.y.tolist()
    tex, tey = te.x.tolist(), te.y.tolist()
    tr_cov, te_cov = tr[covs].values.tolist(), te[covs].values.tolist()
    Xtr, Xte = np.array(tr_cov, float), np.array(te_cov, float)

    rows = []
    for tgt in targets:
        trv, tev = tr[tgt].tolist(), te[tgt].tolist()
        model = gs.fit_variogram(trx, tryy, trv, n_lags=12)
        ok, _ = gs.krige(trx, tryy, trv, model, tex, tey, method="ordinary", max_neighbors=24)
        rk = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov, n_lags=12, max_neighbors=24)
        rf = smelt.RandomForest(n_estimators=300, max_depth=12, seed=seed)
        rf.fit(Xtr, np.array(trv, float))
        rf_tr, rf_te = np.asarray(rf.predict(Xtr)).tolist(), np.asarray(rf.predict(Xte)).tolist()
        hyb = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov,
                                    trend_at_data=rf_tr, trend_at_targets=rf_te,
                                    n_lags=12, max_neighbors=24)
        el = tgt.split("(")[0]
        for method, pred in [("ordinary kriging", ok), ("regression kriging", rk["prediction"]),
                             ("random forest", rf_te), ("RF + residual kriging", hyb["prediction"])]:
            rows.append((el, method, vecv(tev, pred)))
    write_csv(OUT / "multielement.csv", ["element", "method", "vecv"], rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generating figure data (publishable methods only):")
    fig_parity()
    fig_compare()
    fig_idw_tune()
    fig_multielement()
    print("Done. Render with: Rscript figures/figures.R")


if __name__ == "__main__":
    main()
