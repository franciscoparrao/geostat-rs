#!/usr/bin/env python3
"""Generates the data tables for the paper figures (figures/figures.R reads
them). Publishable methods only — no transport/warped kriging.

Outputs to figures/data/:
  parity.csv        geostat-rs vs gstat (OK and IDW) on the Meuse grid
  compare_vecv.csv  leave-one-out VEcv by method on Meuse log-zinc
  idw_tune.csv      IDW power tuning trace (VEcv vs power) on Meuse log-zinc
  multielement.csv  REE hold-out VEcv by element x method, mean/SD over 20
                    deposit-grouped splits (+ random splits as contrast);
                    multielement_meta.json holds n, skew, versions. Needs
                    smelt + data/relaves/tierras_raras.pkl; skipped otherwise.

Run from the repo root, after `Rscript validation/gstat_reference.R` and
`Rscript validation/idw_gstat.R`:
    PYTHONPATH=<dir with geostat_rs.so> python3 figures/make_figure_data.py
"""

import csv
import hashlib
import json
import math
import os
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


def _loo_idw(x, y, v, power, max_neighbors):
    """Manual leave-one-out IDW predictions (per-point, for the bootstrap --
    `compare_methods` only returns the aggregate VEcv/RMSE, not per-point
    arrays)."""
    n = len(x)
    preds = []
    for i in range(n):
        xs = x[:i] + x[i + 1 :]
        ys = y[:i] + y[i + 1 :]
        vs = v[:i] + v[i + 1 :]
        p = gs.idw(xs, ys, vs, [x[i]], [y[i]], power=power, max_neighbors=max_neighbors)
        preds.append(float(p[0]))
    return preds


def _loo_knn(x, y, v, k, max_neighbors):
    """Manual leave-one-out k-NN predictions (see `_loo_idw`)."""
    n = len(x)
    preds = []
    for i in range(n):
        xs = x[:i] + x[i + 1 :]
        ys = y[:i] + y[i + 1 :]
        vs = v[:i] + v[i + 1 :]
        p = gs.knn(xs, ys, vs, [x[i]], [y[i]], k=k)
        preds.append(float(p[0]))
    return preds


def bootstrap_vecv_ci(obs, preds_by_method, n_boot=2000, seed=42):
    """95% CI for VEcv per method via a nonparametric bootstrap: resample
    point indices with replacement (the SAME resample for every method each
    iteration, for a fair paired comparison), recompute VEcv on the
    resampled observed/predicted pairs, and take the 2.5/97.5 percentiles
    of the resulting distribution."""
    rng = random.Random(seed)
    n = len(obs)
    obs = list(obs)
    boot = {m: [] for m in preds_by_method}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        o = [obs[i] for i in idx]
        for m, preds in preds_by_method.items():
            p = [preds[i] for i in idx]
            boot[m].append(vecv(o, p))
    out = {}
    for m, vals in boot.items():
        vals.sort()
        lo = vals[int(0.025 * n_boot)]
        hi = vals[min(int(0.975 * n_boot), n_boot - 1)]
        out[m] = (lo, hi)
    return out


def fig_compare():
    """Leave-one-out VEcv by method on Meuse log-zinc, with a bootstrap 95%
    CI (AUDIT/review follow-up: point-estimate rankings alone do not show
    whether e.g. IDW and k-NN's near-tied VEcv exceeds LOO sampling noise)."""
    x, y, v = meuse()
    cmp = gs.compare_methods(x, y, v, max_neighbors=32, knn_k=8)
    label = {
        "ordinary_kriging": "ordinary kriging",
        "idw": "IDW",
        "knn": "k-NN",
        "nearest_neighbor": "nearest neighbour",
    }

    # Per-point LOO predictions for the bootstrap, matching compare_methods'
    # internal configuration exactly (max_neighbors=32, IDW power 2, k=8/1).
    model = gs.fit_variogram(x, y, v, n_lags=15)
    ok_cv = gs.loo_cv(x, y, v, model, method="ordinary", max_neighbors=32)
    preds_by_method = {
        "ordinary kriging": [float(p) for p in ok_cv["predicted"]],
        "IDW": _loo_idw(x, y, v, power=2.0, max_neighbors=32),
        "k-NN": _loo_knn(x, y, v, k=8, max_neighbors=32),
        "nearest neighbour": _loo_knn(x, y, v, k=1, max_neighbors=32),
    }
    ci = bootstrap_vecv_ci(v, preds_by_method)

    rows = [(label[k], m["vecv"], m["rmse"], *ci[label[k]]) for k, m in cmp.items()]
    rows.sort(key=lambda r: -r[1])
    write_csv(
        OUT / "compare_vecv.csv",
        ["method", "vecv", "rmse", "vecv_lo", "vecv_hi"],
        rows,
    )


def fig_idw_tune():
    """IDW power tuning trace (VEcv vs power)."""
    x, y, v = meuse()
    res = gs.tune_idw_power(x, y, v, powers=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])
    rows = [(p, s) for p, s in res["trace"]]
    write_csv(OUT / "idw_tune.csv", ["power", "vecv"], rows)
    print(f"  best IDW power = {res['best']} (VEcv {res['best_vecv']:.1f})")


def fig_anisotropy():
    """Meuse log-zinc anisotropy: the 2-D variogram map plus directional
    variograms along the fitted major and minor axes with the fitted
    anisotropic model overlaid. Showcases the variogram map and the automatic
    geometric-anisotropy fit on a textbook dataset."""
    x, y, v = meuse()
    diag = math.hypot(max(x) - min(x), max(y) - min(y))
    max_dist = diag / 3.0
    n_lags = 15

    # 2-D variogram map spanning roughly the populated lag region.
    n_map = 12
    lag_width = max_dist / n_map
    vm = gs.variogram_map(x, y, v, n_lags=n_map, lag_width=lag_width)
    rows = [
        (hx, hy, (g if g == g else ""), npr)
        for hx, hy, g, npr in zip(vm["hx"], vm["hy"], vm["gamma"], vm["n_pairs"])
    ]
    write_csv(OUT / "meuse_vmap.csv", ["hx", "hy", "gamma", "n_pairs"], rows)

    # Fit the anisotropic model and read its parameters.
    model = gs.fit_anisotropic(x, y, v, n_dirs=4, n_lags=n_lags, max_dist=max_dist)
    az, ratio = model.anisotropy()
    minor_az = (az + 90.0) % 180.0
    mj = json.loads(model.to_json())
    st = mj["structures"][0]
    write_csv(
        OUT / "meuse_aniso_params.csv",
        ["azimuth", "ratio", "major_range", "minor_range", "nugget", "sill"],
        [(az, ratio, st["range"], st["range"] * ratio, mj["nugget"], st["sill"])],
    )

    # Directional experimental variograms along the major and minor axes.
    dir_rows = []
    for name, a in [("major", az), ("minor", minor_az)]:
        h, gamma, npr = gs.experimental_variogram(
            x, y, v, n_lags=n_lags, max_dist=max_dist, azimuth=a, tolerance=25.0
        )
        for hi, gi, ni in zip(h, gamma, npr):
            if ni > 0 and gi == gi:
                dir_rows.append((name, hi, gi, ni))
    write_csv(OUT / "meuse_aniso_dir.csv", ["axis", "h", "gamma", "n_pairs"], dir_rows)

    # Fitted model curves: major uses gamma(h); minor stretches the lag by the
    # ratio (the minor-axis effective range is ratio*major).
    fit_rows = []
    for i in range(101):
        hi = max_dist * i / 100.0
        fit_rows.append(("major", hi, model.gamma(hi)))
        fit_rows.append(("minor", hi, model.gamma(hi / ratio)))
    write_csv(OUT / "meuse_aniso_fit.csv", ["axis", "h", "gamma"], fit_rows)
    print(f"  fitted anisotropy: azimuth {az:.1f} deg, ratio {ratio:.3f}")


REE_PKL = Path(os.environ.get(
    "GEOSTAT_REE_PKL", Path(__file__).resolve().parents[1] / "data" / "relaves" / "tierras_raras.pkl"))
# Published subset of the registry (data/relaves/export_csv.py); preferred when present.
REE_CSV = Path(__file__).resolve().parents[1] / "data" / "relaves" / "coquimbo_ree.csv"


def load_ree():
    """Registry rows as a DataFrame and the path they came from (CSV first)."""
    import pandas as pd
    if REE_CSV.exists():
        return pd.read_csv(REE_CSV), REE_CSV
    return pd.read_pickle(REE_PKL), REE_PKL


def fig_multielement(reps=20):
    """REE hold-out VEcv by element x method on log grades.

    Data provenance: the public geochemical registry of tailings deposits in
    Chile (SERNAGEOMIN, 2023); see data/relaves/provenance.json.

    Modelling choices (each answers a flaw found in review):
    * Grades and covariates are log-transformed (skew up to 10 on the raw
      scale); VEcv and interval coverage are computed on the log scale. Values
      sitting on a substitution floor (e.g. Y = 5 g/t, a below-detection code)
      are kept as given and counted in the meta file.
    * Variography uses explicit short lags (VARIO_LAGS over VARIO_MAX_M), the
      scale at which the hold-out distances live, not a default of a third of
      the 300-km bounding-box diagonal.
    * The RF + residual-kriging hybrid kriges cross-fitted residuals: the trend
      at each training point comes from a forest that did not see that
      point's deposit (CROSSFIT_K grouped folds inside the training set).
    * Baselines add an RF with coordinates (RFsp-style) to the OK, linear
      regression kriging and covariate-only RF.
    * Three hold-out schemes, `reps` splits each: whole deposits
      (Faena + Deposito, primary), whole operations (Faena, sensitivity) and
      random locations (leaky contrast). ~25% of locations go to test.
    Writes mean/SD per cell, 80%-interval coverage and Deutsch goodness for
    the kriging-based methods, and multielement_meta.json with n, censoring,
    skew, fitted variograms, distances, paired gains and package versions.
    """
    try:
        import numpy as np
        import pandas as pd
        import smelt
    except ImportError as e:
        print(f"  multielement skipped (missing {e.name})")
        return
    if not (REE_CSV.exists() or REE_PKL.exists()):
        print(f"  multielement skipped (dataset not found: {REE_CSV} or {REE_PKL})")
        return

    region = "COQUIMBO"
    targets = ["La(g/t)", "Ce(g/t)", "Nd(g/t)", "Dy(g/t)", "Y(g/t)"]
    covs = ["P2O5(%)", "Th(g/t)", "TiO2(%)", "Fe2O3(%)", "Zr(g/t)"]
    seed = 20260618
    VARIO_LAGS, VARIO_MAX_M = 20, 2000.0
    CROSSFIT_K = 5
    NB = 24
    rf_args = dict(n_estimators=300, max_depth=12, seed=seed)

    def num(s):
        s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s.str.replace("<", "", regex=False).str.replace(">", "", regex=False),
                             errors="coerce")

    def skew(v):
        a = np.asarray(v, float)
        return float(((a - a.mean()) ** 3).mean() / a.std() ** 3)

    d, src = load_ree()
    df = pd.DataFrame({"x": num(d["Coord. E"]), "y": num(d["Coord. N"])})
    for c in targets + covs:
        df[c] = num(d[c])
    df["r"] = d["Region"]
    df["grp"] = d["Faena"].astype(str).str.strip() + "|" + d["Deposito"].astype(str).str.strip()
    df["op"] = d["Faena"].astype(str).str.strip()
    df = df[(df.x > 0) & (df.y > 0) & (df.r == region)].drop(columns="r").dropna()
    n_samples, n_deposits, n_ops = len(df), int(df.grp.nunique()), int(df.op.nunique())
    floor = {}
    for c in targets + covs:
        lo = df[c].min()
        floor[c.split("(")[0]] = {"value": float(lo), "count": int((df[c] == lo).sum()),
                                  "fraction": float((df[c] == lo).mean())}
    raw_skew = {t.split("(")[0]: skew(df[t]) for t in targets}
    for c in targets + covs:
        df[c] = np.log(df[c])
    # Average collocated samples (same location, to the metre) on the log scale.
    agg = {"x": "mean", "y": "mean", "grp": "first", "op": "first",
           **{c: "mean" for c in targets + covs}}
    df = (df.assign(xr=df.x.round(0), yr=df.y.round(0))
            .groupby(["xr", "yr"], as_index=False).agg(agg).reset_index(drop=True))
    n, nt = len(df), len(df) // 4

    def split_random(rep):
        idx = list(range(n))
        random.Random(seed + rep).shuffle(idx)
        return idx[nt:], idx[:nt]

    def grouped(col):
        groups = sorted(df[col].unique())
        sizes = df.groupby(col).size()

        def split(rep):
            g = groups[:]
            random.Random(seed + rep).shuffle(g)
            test, cnt = set(), 0
            for x in g:
                if cnt >= nt:
                    break
                test.add(x)
                cnt += sizes[x]
            mask = df[col].isin(test)
            return df.index[~mask].tolist(), df.index[mask].tolist()
        return split

    def crossfit(tr, X, y, extra=None):
        """Grouped K-fold out-of-deposit RF predictions at the training points."""
        g = sorted(tr.grp.unique())
        random.Random(seed).shuffle(g)
        fold_of = {x: i % CROSSFIT_K for i, x in enumerate(g)}
        folds = tr.grp.map(fold_of).values
        out = np.empty(len(tr))
        for k in range(CROSSFIT_K):
            m = folds == k
            rf = smelt.RandomForest(**rf_args)
            rf.fit(X[~m], y[~m])
            out[m] = np.asarray(rf.predict(X[m]))
        return out

    def model_dict(m):
        j = json.loads(m.to_json())
        st = j.get("structures", [{}])[0]
        return {"nugget": j.get("nugget"), "kind": st.get("kind"), "psill": st.get("sill"),
                "range": st.get("range"), "total_sill": m.total_sill()}

    methods = ["ordinary kriging", "regression kriging", "random forest",
               "random forest + coordinates", "RF + residual kriging"]
    kriged = {"ordinary kriging", "regression kriging", "RF + residual kriging"}
    scores, cover, good = {}, {}, {}
    nn_med, n_test = {}, {}
    schemes = [("deposit-grouped", grouped("grp")), ("operation-grouped", grouped("op")),
               ("random", split_random)]
    for scheme, split in schemes:
        for rep in range(reps):
            tri, tei = split(rep)
            tr, te = df.iloc[tri], df.iloc[tei]
            a, b = te[["x", "y"]].values, tr[["x", "y"]].values
            nn = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
            nn_med.setdefault(scheme, []).append(float(np.median(nn)))
            n_test.setdefault(scheme, []).append(len(te))
            trx, tryy = tr.x.tolist(), tr.y.tolist()
            tex, tey = te.x.tolist(), te.y.tolist()
            tr_cov, te_cov = tr[covs].values.tolist(), te[covs].values.tolist()
            Xtr, Xte = np.array(tr_cov, float), np.array(te_cov, float)
            Xtr_c = np.column_stack([Xtr, tr[["x", "y"]].values])
            Xte_c = np.column_stack([Xte, te[["x", "y"]].values])
            for tgt in targets:
                ytr = tr[tgt].values
                trv, tev = ytr.tolist(), te[tgt].tolist()
                model = gs.fit_variogram(trx, tryy, trv, n_lags=VARIO_LAGS, max_dist=VARIO_MAX_M)
                ok, ok_var = gs.krige(trx, tryy, trv, model, tex, tey, method="ordinary",
                                      max_neighbors=NB)
                rk = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov,
                                           n_lags=VARIO_LAGS, max_dist=VARIO_MAX_M,
                                           max_neighbors=NB)
                rf = smelt.RandomForest(**rf_args)
                rf.fit(Xtr, ytr)
                rf_te = np.asarray(rf.predict(Xte))
                rfc = smelt.RandomForest(**rf_args)
                rfc.fit(Xtr_c, ytr)
                rfc_te = np.asarray(rfc.predict(Xte_c))
                rf_cf = crossfit(tr, Xtr, ytr)
                hyb = gs.regression_kriging(trx, tryy, trv, tr_cov, tex, tey, te_cov,
                                            trend_at_data=rf_cf.tolist(),
                                            trend_at_targets=rf_te.tolist(),
                                            n_lags=VARIO_LAGS, max_dist=VARIO_MAX_M,
                                            max_neighbors=NB)
                el = tgt.split("(")[0]
                preds = {"ordinary kriging": (ok, ok_var),
                         "regression kriging": (rk["prediction"], rk["variance"]),
                         "random forest": (rf_te, None),
                         "random forest + coordinates": (rfc_te, None),
                         "RF + residual kriging": (hyb["prediction"], hyb["variance"])}
                for m in methods:
                    pred, var = preds[m]
                    v = vecv(tev, pred)
                    if not math.isfinite(v):
                        # Fail loudly: a mean over splits must never drop one.
                        raise RuntimeError(f"non-finite VEcv: {scheme} split {rep} {el} {m}")
                    scores.setdefault((scheme, el, m), []).append(v)
                    if m in kriged:
                        sd = np.sqrt(np.maximum(np.asarray(var, float), 0.0))
                        ap = gs.accuracy_plot(tev, list(map(float, pred)), sd.tolist())
                        nom = list(ap["nominal"])
                        i80 = min(range(len(nom)), key=lambda i: abs(nom[i] - 0.8))
                        cover.setdefault((scheme, el, m), []).append(float(ap["observed"][i80]))
                        good.setdefault((scheme, el, m), []).append(float(ap["goodness"]))
        print(f"  multielement: {scheme} done", flush=True)

    rows = []
    for (scheme, el, m), v in scores.items():
        v = np.asarray(v)
        c = cover.get((scheme, el, m))
        g = good.get((scheme, el, m))
        rows.append((scheme, el, m, float(v.mean()), float(v.std(ddof=1)),
                     float(np.mean(c)) if c else "", float(np.mean(g)) if g else "", len(v)))
    write_csv(OUT / "multielement.csv",
              ["scheme", "element", "method", "vecv_mean", "vecv_sd",
               "coverage80_mean", "goodness_mean", "n_splits"], rows)

    # Paired gains and descriptive variography on the full dataset.
    gain = {}
    for scheme, _ in schemes:
        for tgt in targets:
            el = tgt.split("(")[0]
            for other in ("random forest", "regression kriging"):
                diff = (np.asarray(scores[(scheme, el, "RF + residual kriging")])
                        - np.asarray(scores[(scheme, el, other)]))
                gain[f"{scheme}:{el}:hybrid-minus-{other}"] = {
                    "mean": float(diff.mean()), "sd": float(diff.std(ddof=1))}
    xs, ys = df.x.tolist(), df.y.tolist()
    X_all = df[covs].values.astype(float)
    vario = {}
    for tgt in targets:
        el = tgt.split("(")[0]
        yv = df[tgt].values
        rf = smelt.RandomForest(**rf_args)
        rf.fit(X_all, yv)
        res_in = yv - np.asarray(rf.predict(X_all))
        res_cf = yv - crossfit(df, X_all, yv)
        vario[el] = {
            "grade": model_dict(gs.fit_variogram(xs, ys, yv.tolist(), n_lags=VARIO_LAGS,
                                                 max_dist=VARIO_MAX_M)),
            "rf_residual_in_sample": model_dict(gs.fit_variogram(
                xs, ys, res_in.tolist(), n_lags=VARIO_LAGS, max_dist=VARIO_MAX_M)),
            "rf_residual_cross_fitted": model_dict(gs.fit_variogram(
                xs, ys, res_cf.tolist(), n_lags=VARIO_LAGS, max_dist=VARIO_MAX_M)),
            "var_in_sample_residual": float(res_in.var()),
            "var_cross_fitted_residual": float(res_cf.var()),
            "var_grade": float(yv.var()),
        }
    meta = {
        "dataset_file": src.name, "dataset_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "region": region, "n_samples": n_samples, "n_deposits": n_deposits,
        "n_operations": n_ops, "n_locations": n, "scale": "natural log of grade and covariates",
        "raw_skew_by_element": raw_skew, "floor_values": floor,
        "variography": {"n_lags": VARIO_LAGS, "max_dist_m": VARIO_MAX_M,
                        "lag_width_m": VARIO_MAX_M / VARIO_LAGS},
        "crossfit_folds": CROSSFIT_K, "max_neighbors": NB, "rf": rf_args,
        "n_splits": reps, "seed": seed,
        "n_test_median": {k: float(np.median(v)) for k, v in n_test.items()},
        "nearest_train_distance_m_median": {k: float(np.median(v)) for k, v in nn_med.items()},
        "paired_gains": gain, "variograms_full_data": vario,
        "geostat_rs_version": gs.__version__,
        "smelt_version": getattr(smelt, "__version__", "unknown"),
    }
    (OUT / "multielement_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generating figure data (publishable methods only):")
    fig_parity()
    fig_compare()
    fig_idw_tune()
    fig_anisotropy()
    fig_multielement()
    print("Done. Render with: Rscript figures/figures.R")


if __name__ == "__main__":
    main()
