"""
train_all_final.py
==================
The locked-in production model, distilled from all the experiments:

  engineered features (IV + dI/dV + stats, binned) + pH + technique
  -> ExtraTrees vs HGB (auto-pick the better per group)
  -> honest grouped cross-validation
  -> headline metric = MAE / factor error (stable), R2 shown but secondary
  -> per-concentration breakdown + saved model + parity plot

This beat every autoencoder variant (plain AE, conv AE, VAE) at this sample size,
so the autoencoders are dropped. pH and technique are already columns in the
engineered feature matrix, so they reach the regressor directly.

Defaults: drop 0 M blanks (keep >= 5 nM), average the 4 replicate scans,
group CV by physical sample so a sample's DPV/SWV never straddle the split.

Usage:
    python train_all_final.py                  # DA single (default)
    python train_all_final.py DA single
    python train_all_final.py --all            # every group, one summary table
    python train_all_final.py --analytes DA SER --sources single mixture
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import config
from src.data import load_dataset, group_mask, available_groups, format_conc
from src.features import drop_constant_columns
from src.calibration_plot import calibration_plot

warnings.filterwarnings("ignore")


def candidates():
    c = {
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=500, random_state=0, n_jobs=-1),
    }
    c["HGB"] = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                     l2_regularization=1.0, random_state=0)
    try:
        from xgboost import XGBRegressor
        c["XGBoost"] = lambda: XGBRegressor(n_estimators=400, learning_rate=0.05,
                                            max_depth=4, subsample=0.9, colsample_bytree=0.9,
                                            random_state=0, n_jobs=-1, verbosity=0)
    except Exception:
        pass
    return c


def average_replicates(X, y, meta):
    keys = [k for k in ["filename", "technique", "conc_index"] if k in meta.columns]
    meta = meta.reset_index(drop=True)
    X_avg, y_avg, rows = [], [], []
    for _, idx in meta.groupby(keys, sort=False).indices.items():
        idx = np.asarray(idx)
        X_avg.append(X[idx].mean(axis=0)); y_avg.append(y[idx].mean())
        rows.append(meta.iloc[idx[0]])
    return (np.array(X_avg, np.float32), np.array(y_avg, np.float32),
            pd.DataFrame(rows).reset_index(drop=True))


def grouped_cv(model_fn, X, y, groups, folds, strat=None, repeats=10):
    """Robust grouped CV: StratifiedGroupKFold (every concentration level present
    in each training fold) repeated over seeds. Falls back to GroupKFold if no
    stratification labels are given. Returns (r2, r2_std, mae, mae_std, oof) where
    oof is from the first repeat (for plotting/breakdown)."""
    from sklearn.model_selection import StratifiedGroupKFold
    n_groups = len(np.unique(groups))
    if strat is not None:
        per_level = pd.Series(groups).groupby(strat).nunique()
        max_f = int(min(per_level.min(), n_groups))
        f = max(2, min(folds, max_f))
        r2_reps, mae_reps, oof0 = [], [], None
        for r in range(repeats):
            sgkf = StratifiedGroupKFold(n_splits=f, shuffle=True, random_state=r)
            oof = np.full(len(y), np.nan)
            try:
                splits = list(sgkf.split(X, strat, groups))
            except ValueError:
                continue
            r2s, maes = [], []
            for tr, te in splits:
                m = model_fn().fit(X[tr], y[tr])
                p = np.asarray(m.predict(X[te])).ravel()
                oof[te] = p
                r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
            if oof0 is None:
                oof0 = oof
            r2_reps.append(np.mean(r2s)); mae_reps.append(np.mean(maes))
        if mae_reps:
            return (float(np.mean(r2_reps)), float(np.std(mae_reps)),
                    float(np.mean(mae_reps)), float(np.std(mae_reps)), oof0)

    # fallback: plain GroupKFold
    gkf = GroupKFold(min(folds, n_groups))
    r2s, maes, oof = [], [], np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = model_fn().fit(X[tr], y[tr])
        p = np.asarray(m.predict(X[te])).ravel()
        oof[te] = p
        r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
    return np.mean(r2s), np.std(r2s), np.mean(maes), np.std(maes), oof


def run_group(ds, analyte, source, args):
    m = group_mask(ds, analyte, source)
    if m.sum() == 0:
        return None
    X, kept_names, _ = drop_constant_columns(ds.X[m], ds.feature_names)
    y = ds.y[m]
    meta = ds.meta[m].reset_index(drop=True)

    conc = meta["concentration_M"].values.astype(float)
    topk = config.WORKING_RANGE_TOPK.get((analyte, source))
    use_wr = topk is not None and (args.use_working_range
                                   or config.USE_WORKING_RANGE_BY_DEFAULT) \
             and not args.ignore_working_range
    if use_wr:
        keep = config.topk_conc_mask(conc, topk)
        n_levels = len(np.unique(conc[keep]))
        print(f"  {analyte}/{source}: working range = top-{topk} levels "
              f"({n_levels} kept)")
    else:
        keep = conc >= args.min_conc
    X, y, meta = X[keep], y[keep], meta[keep].reset_index(drop=True)
    if len(y) < config.MIN_SAMPLES_PER_GROUP:
        print(f"  {analyte}/{source}: too few samples after filtering, skipped")
        return None

    X, y, meta = average_replicates(X, y, meta)

    # CV groups: physical sample (both techniques together)
    gcols = [c for c in ["filename", "conc_index"] if c in meta.columns]
    groups = meta[gcols].astype(str).agg("|".join, axis=1).values

    print(f"\n{'='*60}\n  {analyte} ({source})  —  {len(y)} measurements, "
          f"{len(np.unique(groups))} samples\n{'='*60}")

    # auto-pick by stable metric (MAE), robust stratified-repeated grouped CV
    strat = meta["conc_index"].values.astype(int) if "conc_index" in meta.columns else None
    scored = {name: grouped_cv(fn, X, y, groups, args.folds, strat=strat)
              for name, fn in candidates().items()}
    best = min(scored.items(), key=lambda kv: kv[1][2])  # lowest MAE
    bname, (r2, r2s, mae, maes, oof) = best

    for name, s in scored.items():
        tag = "  <- best" if name == bname else ""
        print(f"  {name:<12} R2={s[0]:.3f}  MAE={s[2]:.3f} ({10**s[2]:.1f}x){tag}")

    # per-concentration breakdown (robust view)
    print("  per-concentration error (grouped out-of-fold):")
    dfp = pd.DataFrame({"conc": meta["concentration_M"].values, "yt": y, "yp": oof}).dropna()
    for c, g in dfp.groupby("conc"):
        mc = np.mean(np.abs(g["yt"] - g["yp"]))
        print(f"    {format_conc(np.log10(c)):<12}{len(g):>4}   "
              f"MAE={mc:.3f} ({10**mc:.1f}x)")

    # save final model (refit on all group data)
    final = candidates()[bname]().fit(X, y)
    gdir = config.OUTPUT_DIR / f"{analyte}_{source}_final"
    gdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "regressor": bname, "feature_names": kept_names,
                 "min_conc": args.min_conc}, gdir / "model.joblib")
    json.dump({"regressor": bname, "grouped_r2": r2, "grouped_mae": mae,
               "factor_error": 10**mae, "n": int(len(y))},
              open(gdir / "metrics.json", "w"), indent=2)
    _parity(y, oof, f"{analyte} ({source}) — {bname}", r2, mae, gdir / "parity.png")
    calibration_plot(y, oof, f"{analyte} ({source}) — {bname}",
                     gdir / "parity_calibration.png", fmt_conc=format_conc)

    return {"analyte": analyte, "source": source, "n": len(y), "regressor": bname,
            "r2": round(r2, 3), "mae": round(mae, 3), "factor": f"{10**mae:.1f}x"}


def _parity(yt, yp, label, r2, mae, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = ~np.isnan(yp); yt, yp = yt[ok], yp[ok]
    lims = [min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, color="#2C2C2A", alpha=0.6, s=25)
    ax.plot(lims, lims, "--", color="#A32D2D", lw=1.5)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("True log10(M)"); ax.set_ylabel("Predicted log10(M)")
    ax.set_title(f"{label}\ngrouped R2={r2:.3f}  MAE={mae:.3f} ({10**mae:.1f}x)")
    ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(fig)


def _calibration(yt, yp, conc, label, out):
    """Calibration view: one point per TRUE concentration level = mean predicted
    log10, with error bars = MAE at that level (typical per-sample miss). Faint
    raw points shown behind for honesty. A point near the diagonal with a long
    bar means predictions average out correct but individual misses are large."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = ~np.isnan(yp)
    yt, yp, conc = yt[ok], yp[ok], conc[ok]
    levels = sorted(np.unique(yt))
    means, maes, ns = [], [], []
    for lv in levels:
        sel = yt == lv
        means.append(float(np.mean(yp[sel])))
        maes.append(float(np.mean(np.abs(yp[sel] - lv))))   # MAE at this level
        ns.append(int(sel.sum()))
    means, maes = np.array(means), np.array(maes)

    lims = [min(min(levels), yp.min()) - 0.3, max(max(levels), yp.max()) + 0.3]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, color="#BBBBBB", alpha=0.35, s=15, zorder=1)        # faint raw
    ax.errorbar(levels, means, yerr=maes, fmt="o", color="#2C2C2A",
                ecolor="#2C2C2A", capsize=5, ms=7, lw=1.5, zorder=3,
                label="mean prediction ± MAE")
    ax.plot(lims, lims, "--", color="#A32D2D", lw=1.5, zorder=2, label="ideal")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("True log10(M)"); ax.set_ylabel("Predicted log10(M)")
    overall_mae = float(np.mean(np.abs(yp - yt)))
    ax.set_title(f"{label} — calibration\nbars = MAE per level (overall {overall_mae:.2f}, "
                 f"{10**overall_mae:.1f}x)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analyte", nargs="?", default="DA")
    ap.add_argument("source", nargs="?", default="single")
    ap.add_argument("--all", action="store_true", help="run every available group")
    ap.add_argument("--analytes", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bin-size", type=int, default=None,
                    help="override config.BIN_SIZE for this run")
    ap.add_argument("--min-conc", type=float, default=5e-9)
    ap.add_argument("--use-working-range", action="store_true",
                    help="apply config.WORKING_RANGE_TOPK per group")
    ap.add_argument("--ignore-working-range", action="store_true",
                    help="force --min-conc floor, ignore the working-range map")
    args = ap.parse_args()
    if args.bin_size:
        config.BIN_SIZE = args.bin_size
        try:
            load_dataset.cache_clear()
        except Exception:
            pass
        print(f'[bin-size override] BIN_SIZE = {config.BIN_SIZE}')

    ds = load_dataset()
    if args.all or args.analytes or args.sources:
        if args.analytes:
            config.ANALYTES_TO_TRAIN = [a.upper() for a in args.analytes]
        if args.sources:
            config.SOURCES_TO_TRAIN = [s.lower() for s in args.sources]
        groups = available_groups(ds)
    else:
        groups = [(args.analyte.upper(), args.source.lower())]

    rows = [r for a, s in groups if (r := run_group(ds, a, s, args))]
    if rows:
        summary = pd.DataFrame(rows)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(config.OUTPUT_DIR / "final_summary.csv", index=False)
        print(f"\n{'='*60}\nFINAL SUMMARY (headline = MAE / factor error)\n{'='*60}")
        print(summary.to_string(index=False))
        print(f"\nArtifacts in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()