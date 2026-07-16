"""
data.py
=======
Load the voltammetry pickle straight from the Google Drive desktop folder,
clean it exactly as the notebook did, and build the feature matrix.

The result is cached in memory so repeated group-training in one run only
reads + featurizes the dataset once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

import config
from .gdrive import resolve_dataset_path
from .features import build_features, analyte_from_group


@dataclass
class Dataset:
    X: np.ndarray              # (n, n_features) full feature matrix
    y: np.ndarray             # (n,) log10(concentration M)
    feature_names: list[str]
    analyte_labels: np.ndarray  # (n,) e.g. 'DA','SER'
    source_labels: np.ndarray   # (n,) e.g. 'single','mixture'
    V: np.ndarray
    dx: float
    meta: "pd.DataFrame" = None  # per-row metadata for grouping (filename, technique, ...)
    current_raw: np.ndarray = None  # (n, L) truncated raw current traces (for the conv-AE)


def format_conc(log10_val: float) -> str:
    """log10(M) -> human-readable concentration string (from the notebook)."""
    mol = 10 ** log10_val
    if mol < 1e-12:
        return "blank"
    elif mol < 1e-9:
        return f"{mol/1e-12:.4g} pM"
    elif mol < 1e-6:
        return f"{mol/1e-9:.4g} nM"
    elif mol < 1e-3:
        return f"{mol/1e-6:.4g} uM"
    elif mol < 1:
        return f"{mol/1e-3:.4g} mM"
    return f"{mol:.4g} M"


@lru_cache(maxsize=1)
def load_dataset() -> Dataset:
    """Read, clean, and featurize the dataset (cached for the process lifetime)."""
    path = resolve_dataset_path(config.DATASET_RELATIVE_PATH, config.DRIVE_ROOT)
    print(f"Loading dataset from:\n  {path}")
    if str(path).lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    print(f"Raw shape: {df.shape} | columns: {df.columns.tolist()}")

    # ── Drop short traces ────────────────────────────────────────────────────
    mask = np.array([len(arr) >= config.MIN_TRACE_LENGTH for arr in df["current"].values])
    print(f"Keeping {mask.sum()} / {len(mask)} traces "
          f"(dropping {(~mask).sum()} shorter than {config.MIN_TRACE_LENGTH} pts)")
    df = df[mask].reset_index(drop=True)

    # ── Truncate to common length ────────────────────────────────────────────
    min_len = min(len(arr) for arr in df["current"].values)
    current = np.stack([a[:min_len] for a in df["current"].values]).astype(np.float32)
    voltage = np.stack([a[:min_len] for a in df["voltage"].values]).astype(np.float32)
    V = voltage[0]
    dx = float(V[1] - V[0])

    # ── Target: log10(concentration) ─────────────────────────────────────────
    y_raw = df["concentration_M"].values.astype(float)
    y = np.log10(np.where(y_raw == 0, config.ZERO_CONC_FLOOR, y_raw)).astype(np.float32)

    # ── Metadata ─────────────────────────────────────────────────────────────
    ph_data = df["ph"].values.astype(np.float32)
    technique_data = df["technique"].values
    group_data = df["group"].values
    source_data = df["source"].values

    X, feature_names = build_features(
        current, V, dx, ph_data, technique_data, group_data, source_data,
        bin_size=config.BIN_SIZE,
    )

    analyte_labels = np.array([analyte_from_group(g) for g in group_data])
    # normalize source labels to 'single' / 'mixture'
    source_labels = np.array(
        ["mixture" if "mix" in str(s).lower() else "single" for s in source_data]
    )

    # Metadata for leakage-safe grouping (replicate scans share these keys).
    meta_cols = [c for c in ["filename", "technique", "conc_index", "channel",
                             "ph", "concentration_M", "scan", "title"]
                 if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)

    print(f"Feature matrix: {X.shape} | log10(M) range: {y.min():.2f}..{y.max():.2f}")
    print(f"Analytes: {dict(zip(*np.unique(analyte_labels, return_counts=True)))}")
    print(f"Sources : {dict(zip(*np.unique(source_labels, return_counts=True)))}")

    return Dataset(X, y, feature_names, analyte_labels, source_labels, V, dx, meta, current)


def group_mask(ds: Dataset, analyte: str, source: str) -> np.ndarray:
    """Boolean row mask selecting one analyte+source subset."""
    return (ds.analyte_labels == analyte) & (ds.source_labels == source)


def subset_for_group(ds: Dataset, analyte: str, source: str):
    """Return (X_sub, y_sub, n) for one analyte+source combination."""
    m = group_mask(ds, analyte, source)
    return ds.X[m], ds.y[m], int(m.sum())


def available_groups(ds: Dataset) -> list[tuple[str, str]]:
    """List (analyte, source) combos present in the data, filtered by config."""
    analytes = config.ANALYTES_TO_TRAIN or sorted(set(ds.analyte_labels))
    sources = config.SOURCES_TO_TRAIN or sorted(set(ds.source_labels))
    groups = []
    for a in analytes:
        for s in sources:
            _, _, n = subset_for_group(ds, a, s)
            if n >= config.MIN_SAMPLES_PER_GROUP:
                groups.append((a, s))
    return groups
