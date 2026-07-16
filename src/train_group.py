"""
train_group.py
==============
Train (and optionally tune) ONE specialized model for a single analyte+source
group, then save everything needed for reproducible inference:

    artifacts/<analyte>_<source>/
        model.keras          trained Keras model
        scaler.joblib        the MinMaxScaler fit on this group's training set
        meta.json            kept-feature mask, hyperparameters, metrics

Each group gets its own scaler and (optionally) its own tuned hyperparameters,
so groups never share statistics — this is the whole point of specializing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

import config
from .data import Dataset, subset_for_group
from .features import drop_constant_columns
from .model import build_asymmetric_autoencoder, make_tunable_builder


def group_dir(analyte: str, source: str) -> Path:
    d = config.OUTPUT_DIR / f"{analyte}_{source}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _callbacks():
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_regression_loss", mode="min",
            patience=config.EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_regression_loss", mode="min", factor=0.5,
            patience=config.LR_PLATEAU_PATIENCE, min_lr=1e-6, verbose=0),
    ]


def _tune_hp(input_dim, X_tr, y_tr, X_te, y_te, analyte, source):
    """Run a keras-tuner RandomSearch and return the best hyperparameter dict."""
    import tensorflow as tf
    import keras_tuner as kt

    tuner = kt.RandomSearch(
        make_tunable_builder(input_dim),
        objective=kt.Objective("val_regression_loss", direction="min"),
        max_trials=config.TUNE_MAX_TRIALS,
        executions_per_trial=1,
        directory=str(config.OUTPUT_DIR / "_tuner"),
        project_name=f"tune_{analyte}_{source}",
        overwrite=True,
    )
    tuner.search(
        X_tr, {"reconstruction": X_tr, "regression": y_tr},
        validation_data=(X_te, {"reconstruction": X_te, "regression": y_te}),
        epochs=config.TUNE_EPOCHS, batch_size=config.BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_regression_loss", mode="min",
            patience=config.TUNE_PATIENCE, restore_best_weights=True)],
        verbose=0,
    )
    best = tuner.get_best_hyperparameters(num_trials=1)[0]
    return {k: best.get(k) for k in best.values}


def train_group(ds: Dataset, analyte: str, source: str, tune: bool = True) -> dict:
    """Train one specialized model. Returns a metrics/summary dict."""
    import tensorflow as tf

    X_sub, y_sub, n = subset_for_group(ds, analyte, source)
    label = f"{analyte} ({source})"
    print("\n" + "=" * 60)
    print(f"  Specialized model: {label}  —  {n} samples")
    print("=" * 60)

    if n < config.MIN_SAMPLES_PER_GROUP:
        print(f"  Skipped: fewer than {config.MIN_SAMPLES_PER_GROUP} samples.")
        return {"analyte": analyte, "source": source, "samples": n, "status": "skipped"}

    # Drop columns that are constant within this group (e.g. analyte/source one-hots)
    if config.DROP_CONSTANT_FEATURES:
        X_sub, kept_names, keep_mask = drop_constant_columns(X_sub, ds.feature_names)
    else:
        kept_names, keep_mask = ds.feature_names, np.ones(X_sub.shape[1], bool)
    print(f"  Features used: {X_sub.shape[1]} / {len(ds.feature_names)}")

    # Split + scale (scaler is fit on this group ONLY)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y_sub, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    input_dim = X_tr_sc.shape[1]

    # Hyperparameters: tuned per group, or shared defaults
    if tune:
        print("  Tuning hyperparameters (keras-tuner RandomSearch)...")
        hp = _tune_hp(input_dim, X_tr_sc, y_tr, X_te_sc, y_te, analyte, source)
        print("  Best HP:", {k: hp[k] for k in ("latent_dim", "learning_rate", "reg_weight")})
    else:
        hp = dict(config.DEFAULT_HP)

    # Train the final model
    model, _ = build_asymmetric_autoencoder(input_dim, hp)
    model.fit(
        X_tr_sc, {"reconstruction": X_tr_sc, "regression": y_tr},
        validation_data=(X_te_sc, {"reconstruction": X_te_sc, "regression": y_te}),
        epochs=config.FINAL_EPOCHS, batch_size=config.BATCH_SIZE,
        callbacks=_callbacks(), verbose=0,
    )

    # Evaluate
    _, y_pred = model.predict(X_te_sc, verbose=0)
    y_pred = y_pred.flatten()
    r2 = float(r2_score(y_te, y_pred))
    mae = float(mean_absolute_error(y_te, y_pred))
    print(f"  R2  = {r2:.4f}")
    print(f"  MAE = {mae:.4f} log10(M)   (concentration factor error ~{10**mae:.1f}x)")

    # Save artifacts
    gdir = group_dir(analyte, source)
    model.save(gdir / "model.keras")
    joblib.dump(scaler, gdir / "scaler.joblib")
    meta = {
        "analyte": analyte, "source": source, "samples": n,
        "input_dim": input_dim, "tuned": tune,
        "hyperparameters": hp,
        "keep_mask": keep_mask.tolist(),
        "kept_feature_names": kept_names,
        "bin_size": config.BIN_SIZE,
        "metrics": {"r2": r2, "mae_log10": mae, "factor_error": 10 ** mae},
    }
    with open(gdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Parity plot
    _parity_plot(y_te, y_pred, label, r2, mae, gdir / "parity.png")

    return {"analyte": analyte, "source": source, "samples": n,
            "r2": round(r2, 4), "mae": round(mae, 4),
            "factor_error": f"~{10**mae:.1f}x", "status": "ok"}


def _parity_plot(y_true, y_pred, label, r2, mae, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lims = [min(y_true.min(), y_pred.min()) - 0.3, max(y_true.max(), y_pred.max()) + 0.3]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, color="#2C2C2A", alpha=0.6, s=25, linewidths=0)
    ax.plot(lims, lims, color="#A32D2D", lw=1.5, ls="--", label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ticks = np.arange(np.floor(lims[0]), np.ceil(lims[1]) + 1, dtype=int)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"$10^{{{v}}}$" for v in ticks])
    ax.set_yticklabels([f"$10^{{{v}}}$" for v in ticks])
    ax.set_xlabel("True concentration (M)")
    ax.set_ylabel("Predicted concentration (M)")
    ax.set_title(f"{label}\nR2 = {r2:.4f}  |  MAE = {mae:.4f}")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)
