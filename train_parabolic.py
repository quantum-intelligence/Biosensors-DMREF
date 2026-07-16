"""
train_parabolic.py
==================
Parabolic-baseline + fixed-center Gaussian peak features (ported from
Peakfit_DPV/SWV.ipynb), tested honestly against the engineered baseline, plus the
two-stage classify-then-quantify cascade from Process_ML.ipynb.

Feature sets compared (identical stratified-repeated grouped folds):
  engineered        the existing ~170-col set
  parabolic         the 24 Gaussian params (amp/center/width x 8 fixed centers)
  engineered+parab  both

Then the CASCADE (Process_ML.ipynb):
  stage 1: classifier decides if the sample is above a threshold concentration TC
  stage 2: regressor quantifies ONLY the samples predicted above TC
  reported: MAE on the retained samples, and the RETENTION rate (what fraction of
  samples the model is willing to quantify). This is how the published method
  avoids being graded on signals below the detection limit.

The fit runs on RAW traces (the parabola models the background), so point config
at the raw dataset (voltammetry_dataset_aligned.pkl), not the baseline-subtracted one.

Usage:
    python train_parabolic.py DA single
    python train_parabolic.py EP single --tc 5e-7
    python train_parabolic.py DA single --no-cascade
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

import config
from src.data import load_dataset, group_mask, format_conc
from src.features import drop_constant_columns
from src.parabolic_fit import fit_all, feature_names
from src.calibration_plot import calibration_plot

warnings.filterwarnings("ignore")


def et():
    return ExtraTreesRegressor(n_estimators=500, random_state=0, n_jobs=-1)


def etc():
    return ExtraTreesClassifier(n_estimators=500, random_state=0, n_jobs=-1)


def avg_reps(mats, y, meta):
    keys = [k for k in ["filename", "technique", "conc_index"] if k in meta.columns]
    meta = meta.reset_index(drop=True)
    out = {k: [] for k in mats}; ya, rows = [], []
    for _, gi in meta.groupby(keys, sort=False).indices.items():
        gi = np.asarray(gi)
        for k, M in mats.items():
            out[k].append(M[gi].mean(axis=0))
        ya.append(y[gi].mean()); rows.append(meta.iloc[gi[0]])
    return {k: np.array(v, np.float32) for k, v in out.items()}, \
        np.array(ya, np.float32), pd.DataFrame(rows).reset_index(drop=True)


def folds_for(strat, groups, want):
    per = pd.Series(groups).groupby(strat).nunique()
    return max(2, min(want, int(min(per.min(), len(np.unique(groups))))))


def rep_cv(X, y, strat, groups, folds, repeats):
    f = folds_for(strat, groups, folds)
    maes, oof0 = [], None
    for r in range(repeats):
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(f, shuffle=True, random_state=r).split(X, strat, groups):
            oof[te] = et().fit(X[tr], y[tr]).predict(X[te])
        ok = ~np.isnan(oof)
        maes.append(mean_absolute_error(y[ok], oof[ok]))
        if r == 0:
            oof0 = oof
    return float(np.mean(maes)), float(np.std(maes)), oof0


def cascade(X, y, strat, groups, tc_log, folds, repeats):
    """Two-stage: classify above/below TC, then regress only the retained ones."""
    f = folds_for(strat, groups, folds)
    maes, rets, accs = [], [], []
    for r in range(repeats):
        yt_keep, yp_keep, n_tot = [], [], 0
        acc = []
        for tr, te in StratifiedGroupKFold(f, shuffle=True, random_state=r).split(X, strat, groups):
            above_tr = y[tr] >= tc_log
            if above_tr.all() or (~above_tr).all():
                continue                      # degenerate: TC outside this fold's range
            clf = etc().fit(X[tr], above_tr)
            pred_above_te = clf.predict(X[te])
            acc.append(accuracy_score(y[te] >= tc_log, pred_above_te))
            # stage 2: train regressor on training samples PREDICTED above TC
            pred_above_tr = clf.predict(X[tr])
            if pred_above_tr.sum() < 5:
                continue
            reg = et().fit(X[tr][pred_above_tr], y[tr][pred_above_tr])
            n_tot += len(te)
            if pred_above_te.sum():
                yp_keep.extend(reg.predict(X[te][pred_above_te]))
                yt_keep.extend(y[te][pred_above_te])
        if n_tot and yt_keep:
            maes.append(mean_absolute_error(yt_keep, yp_keep))
            rets.append(len(yt_keep) / n_tot)
            accs.append(np.mean(acc))
    if not maes:
        return None
    return (float(np.mean(maes)), float(np.std(maes)),
            float(np.mean(rets)), float(np.mean(accs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analyte"); ap.add_argument("source")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--min-conc", type=float, default=0.0)
    ap.add_argument("--tc", type=float, default=None,
                    help="threshold concentration (M) for the cascade; default = 3rd level")
    ap.add_argument("--no-cascade", action="store_true")
    ap.add_argument("--errbar", choices=["se", "std", "mae"], default="se",
                    help="error bars on the level-averaged parity plots")
    args = ap.parse_args()

    analyte, source = args.analyte.upper(), args.source.lower()
    ds = load_dataset()
    m = group_mask(ds, analyte, source)
    Xeng, _, _ = drop_constant_columns(ds.X[m], ds.feature_names)
    y = ds.y[m]; meta = ds.meta[m].reset_index(drop=True)
    traces = ds.current_raw[m].astype(float)
    tech = meta["technique"].astype(str).values

    print(f"\nFitting parabola + 8 fixed-center Gaussians to {len(traces)} raw traces "
          f"(this takes a minute)...")
    Xpar, ok, rms = fit_all(traces, ds.V, tech)
    print(f"  fit succeeded on {ok.mean()*100:.1f}% of traces "
          f"(median residual RMS {np.nanmedian(rms):.3f})")

    _conc = meta["concentration_M"].values.astype(float)
    sel = (_conc > 0) & (_conc >= args.min_conc) & ok   # always drop 0 M blank
    Xeng, Xpar, y, meta = Xeng[sel], Xpar[sel], y[sel], meta[sel].reset_index(drop=True)

    mats, ya, meta2 = avg_reps({"eng": Xeng, "par": Xpar}, y, meta)
    strat = meta2["conc_index"].values.astype(int)
    gcols = [c for c in ["filename", "conc_index"] if c in meta2.columns]
    groups = meta2[gcols].astype(str).agg("|".join, axis=1).values

    print(f"\n{'='*64}\n  {analyte} ({source}) — {len(ya)} measurements, "
          f"{len(np.unique(groups))} samples\n{'='*64}")

    variants = {
        "engineered": mats["eng"],
        "parabolic (24 gauss)": mats["par"],
        "engineered+parabolic": np.concatenate([mats["eng"], mats["par"]], axis=1),
    }
    res = {}
    for name, X in variants.items():
        mu, sd, oof = rep_cv(X, ya, strat, groups, args.folds, args.repeats)
        res[name] = (mu, sd, oof)
        print(f"  {name:<22} MAE={mu:.3f}+/-{sd:.3f} ({10**mu:.1f}x)")
    best = min(res, key=lambda k: res[k][0])
    d = res["engineered"][0] - res[best][0]
    print(f"  -> best: {best}"
          f"{f' (beats engineered by dMAE {d:.3f})' if best != 'engineered' and d > 0.01 else ' (no gain over engineered)'}")

    # per-concentration: engineered vs best
    print("\n  per-concentration factor error:")
    dfp = pd.DataFrame({"conc": meta2["concentration_M"].values, "yt": ya,
                        "eng": res["engineered"][2], "best": res[best][2]})
    for c, g in dfp.groupby("conc"):
        me = np.mean(np.abs(g["yt"] - g["eng"])); mb = np.mean(np.abs(g["yt"] - g["best"]))
        print(f"    {format_conc(np.log10(c)):<12}{len(g):>4}   "
              f"eng {10**me:5.1f}x  ->  {best.split()[0]} {10**mb:5.1f}x")

    # ---- two-stage cascade (Process_ML.ipynb) ----
    if not args.no_cascade:
        levels = sorted(np.unique(meta2["concentration_M"].values.astype(float)))
        tc = args.tc if args.tc else levels[min(2, len(levels) - 1)]
        tc_log = float(np.log10(tc))
        print(f"\n  {'-'*60}\n  CASCADE (classify >= {format_conc(tc_log)}, then quantify)\n  {'-'*60}")
        for name, X in variants.items():
            out = cascade(X, ya, strat, groups, tc_log, args.folds, args.repeats)
            if out is None:
                print(f"  {name:<22} (TC outside fold range — skipped)")
                continue
            mu, sd, ret, acc = out
            print(f"  {name:<22} MAE={mu:.3f}+/-{sd:.3f} ({10**mu:.1f}x) on "
                  f"{ret*100:.0f}% retained | gate acc {acc*100:.0f}%")
        print("\n  (Cascade MAE is on RETAINED samples only — compare together with"
              "\n   retention: quantifying fewer samples more accurately is the tradeoff.)")

    gdir = config.OUTPUT_DIR / f"{analyte}_{source}_parabolic"
    gdir.mkdir(parents=True, exist_ok=True)
    _parity(ya, res["engineered"][2], f"{analyte} ({source}) engineered",
            res["engineered"][0], gdir / "parity_engineered.png")
    _parity(ya, res[best][2], f"{analyte} ({source}) {best}",
            res[best][0], gdir / "parity_best.png")

    # level-averaged parity (mean +/- SE) for every variant
    fname = {"engineered": "calib_engineered.png",
             "parabolic (24 gauss)": "calib_parabolic.png",
             "engineered+parabolic": "calib_engineered_parabolic.png"}
    for name, (mu, sd, oof) in res.items():
        calibration_plot(ya, oof, f"{analyte} ({source}) — {name}",
                         gdir / fname[name], fmt_conc=format_conc)

    pd.DataFrame([{"variant": k, "MAE": round(v[0], 3), "std": round(v[1], 3),
                   "factor": round(10 ** v[0], 2)} for k, v in res.items()]) \
        .to_csv(gdir / "summary.csv", index=False)
    print(f"\n  saved parity_*.png, {', '.join(fname.values())}, summary.csv to {gdir.name}/")


def _parity(yt, yp, label, mae, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = ~np.isnan(yp); yt, yp = yt[ok], yp[ok]
    r2 = r2_score(yt, yp)
    lims = [min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, color="#2C2C2A", alpha=0.6, s=25)
    ax.plot(lims, lims, "--", color="#A32D2D", lw=1.5)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("True log10(M)"); ax.set_ylabel("Predicted log10(M)")
    ax.set_title(f"{label}\nR2={r2:.3f}  MAE={mae:.3f} ({10**mae:.1f}x)")
    ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()