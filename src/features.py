"""
features.py
===========
Feature engineering carried over from the notebook (Section 2.2):

    [ binned I trace ] + [ binned dI/dV trace ] + [ 10 scalar stats ]
    + [ technique one-hot ] + [ analyte one-hot ] + [ source one-hot ]

The scalar stats are computed on the *un-binned* traces; the I and dI/dV curves
are averaged into bins of `bin_size` to denoise and reduce dimensionality.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks, peak_widths, savgol_filter

# numpy >= 2.0 renamed trapz -> trapezoid (and 2.4 removed trapz). Support both.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

ANALYTE_MAP = {"DA": [1, 0, 0, 0], "EP": [0, 1, 0, 0],
               "NE": [0, 0, 1, 0], "SER": [0, 0, 0, 1]}

SCALAR_NAMES = ["auc", "std", "skew", "didv_max", "didv_min",
                "peak_current", "peak_width", "midpoint", "signal_baseline_ratio", "ph"]


def analyte_from_group(group_label) -> str:
    """'pH4_DA' -> 'DA'. Matches the notebook's parsing exactly."""
    return str(group_label).strip().split("_")[-1].upper()


def build_features(current_data, V, dx, ph_data, technique_data,
                   group_data, source_data, bin_size: int = 3):
    """
    Build the full feature matrix.

    Returns
    -------
    X : (n_samples, n_features) float32
    feature_names : list[str]  (length n_features)
    """
    current_data = np.asarray(current_data, dtype=np.float32)
    didv = np.array([np.gradient(c, dx) for c in current_data], dtype=np.float32)

    # ── Bin the I and dI/dV traces ───────────────────────────────────────────
    n_pts = current_data.shape[1]
    n_bins = n_pts // bin_size
    trim = n_bins * bin_size
    I_binned = current_data[:, :trim].reshape(-1, n_bins, bin_size).mean(axis=2)
    dV_binned = didv[:, :trim].reshape(-1, n_bins, bin_size).mean(axis=2)

    # ── Technique one-hot (DPV / SWV) ────────────────────────────────────────
    tech_onehot = np.array(
        [[1, 0] if t == "DPV" else [0, 1] for t in technique_data], dtype=np.float32
    )

    # ── Analyte one-hot ──────────────────────────────────────────────────────
    analyte_onehot = np.array(
        [ANALYTE_MAP.get(analyte_from_group(g), [0, 0, 0, 0]) for g in group_data],
        dtype=np.float32,
    )

    # ── Source one-hot (mixture / single) ────────────────────────────────────
    source_onehot = np.array(
        [[1, 0] if "mix" in str(s).lower() else [0, 1] for s in source_data],
        dtype=np.float32,
    )

    # ── Scalar features (on un-binned traces) ────────────────────────────────
    scalars = []
    for i in range(len(current_data)):
        I = current_data[i]
        dI = didv[i]
        auc = _trapz(I, dx=dx)
        std = np.std(I)
        skw = sp_stats.skew(I)
        mx, mn = np.max(dI), np.min(dI)
        pks, _ = find_peaks(I, prominence=np.ptp(I) * 0.05)
        if len(pks) > 0:
            mp = pks[np.argmax(I[pks])]
            pkc = I[mp]
            pkw = peak_widths(I, [mp], rel_height=0.5)[0][0] * dx
        else:
            pkc = np.max(I)
            pkw = 0.0
        q2 = I[len(I) // 2]
        sbr = pkc / (np.mean(I) + 1e-12)
        ph = ph_data[i]
        scalars.append([auc, std, skw, mx, mn, pkc, pkw, q2, sbr, ph])

    sc = np.nan_to_num(np.array(scalars, dtype=np.float32))

    X = np.concatenate(
        [I_binned, dV_binned, sc, tech_onehot, analyte_onehot, source_onehot], axis=1
    ).astype(np.float32)

    feature_names = (
        [f"I_bin{i}" for i in range(n_bins)]
        + [f"didv_bin{i}" for i in range(n_bins)]
        + SCALAR_NAMES
        + ["tech_DPV", "tech_SWV"]
        + ["analyte_DA", "analyte_EP", "analyte_NE", "analyte_SER"]
        + ["source_mixture", "source_single"]
    )
    return X, feature_names


PEAK_NAMES = ["peak_height", "peak_area", "peak_potential", "fwhm",
              "prominence", "left_slope", "right_slope", "baseline_slope",
              "auc_total", "ph"]


def peak_features(current_data, V, ph_data, technique_data, smooth: int = 7):
    """
    Compact, physically-grounded descriptors of each trace's dominant oxidation
    peak (auto-detected). Far fewer, stronger features than the binned trace.

    For each trace: detect the main peak, draw a linear baseline between its two
    feet, and measure baseline-corrected height, area, potential, width, and the
    flanking slopes. Adds pH and a technique one-hot.

    Returns (X_peak, feature_names).
    """
    current_data = np.asarray(current_data, dtype=np.float64)
    n, L = current_data.shape
    feats = np.zeros((n, len(PEAK_NAMES)), dtype=np.float64)
    dV = abs(V[1] - V[0])

    win = smooth if smooth % 2 == 1 else smooth + 1
    for i in range(n):
        I = current_data[i]
        ph = ph_data[i]
        Is = savgol_filter(I, win, 2) if L >= win + 2 else I
        rng = np.ptp(Is) + 1e-12
        pks, props = find_peaks(Is, prominence=rng * 0.05)

        baseline_slope = (I[-1] - I[0]) / (V[-1] - V[0] + 1e-12)
        auc_total = _trapz(I - np.linspace(I[0], I[-1], L), dx=dV)

        if len(pks) == 0:
            mp = int(np.argmax(Is))
            height = Is[mp] - np.interp(V[mp], [V[0], V[-1]], [I[0], I[-1]])
            feats[i] = [height, 0.0, V[mp], 0.0, 0.0, 0.0, 0.0,
                        baseline_slope, auc_total, ph]
            continue

        j = int(np.argmax(props["prominences"]))
        main = int(pks[j])
        lb = max(0, int(props["left_bases"][j]))
        rb = min(L - 1, int(props["right_bases"][j]))
        if rb <= lb:
            lb, rb = max(0, main - 1), min(L - 1, main + 1)

        base = np.interp(np.arange(L), [lb, rb], [I[lb], I[rb]])
        corr = I - base
        height = float(corr[main])
        area = float(_trapz(np.clip(corr[lb:rb + 1], 0, None), dx=dV))
        potential = float(V[main])
        try:
            w = peak_widths(np.clip(corr, 0, None), [main], rel_height=0.5)[0][0]
            fwhm = float(w * dV)
        except Exception:
            fwhm = 0.0
        prominence = float(props["prominences"][j])
        left_slope = float((I[main] - I[lb]) / (V[main] - V[lb] + 1e-12))
        right_slope = float((I[rb] - I[main]) / (V[rb] - V[main] + 1e-12))
        feats[i] = [height, area, potential, fwhm, prominence,
                    left_slope, right_slope, baseline_slope, auc_total, ph]

    feats = np.nan_to_num(feats).astype(np.float32)
    tech_onehot = np.array(
        [[1, 0] if t == "DPV" else [0, 1] for t in technique_data], dtype=np.float32)
    X = np.concatenate([feats, tech_onehot], axis=1)
    names = list(PEAK_NAMES) + ["tech_DPV", "tech_SWV"]
    return X, names


PH_COUPLED_NAMES = ["peak_potential", "peak_potential_pHcorr", "pH_centered",
                    "pHxheight", "pHxarea", "pHxpeakpot"]


def ph_coupled_features(current_data, V, ph_data, ref_ph: float = 7.0,
                        mv_per_ph: float = 0.059, smooth: int = 7):
    """
    Features that encode the proton-coupling of catecholamine oxidation.

    Catecholamine oxidation is proton-coupled, so the peak potential shifts with
    pH (~59 mV per pH unit, the Nernstian slope). These columns make that
    relationship explicit instead of leaving the model to infer it:

      peak_potential          dominant-peak position (V)
      peak_potential_pHcorr   peak position corrected to ref_ph (pH-invariant)
      pH_centered             pH - ref_ph
      pHxheight / pHxarea / pHxpeakpot   pH x signal interactions

    Returns (X_aug, names). Concatenate onto the engineered features.
    """
    current_data = np.asarray(current_data, dtype=np.float64)
    n, L = current_data.shape
    out = np.zeros((n, len(PH_COUPLED_NAMES)), dtype=np.float64)
    dV = abs(V[1] - V[0])
    win = smooth if smooth % 2 == 1 else smooth + 1

    for i in range(n):
        I = current_data[i]
        ph = float(ph_data[i])
        Is = savgol_filter(I, win, 2) if L >= win + 2 else I
        pks, props = find_peaks(Is, prominence=(np.ptp(Is) + 1e-12) * 0.05)
        if len(pks) == 0:
            main = int(np.argmax(Is)); lb, rb = 0, L - 1
        else:
            j = int(np.argmax(props["prominences"]))
            main = int(pks[j])
            lb = max(0, int(props["left_bases"][j])); rb = min(L - 1, int(props["right_bases"][j]))
            if rb <= lb:
                lb, rb = max(0, main - 1), min(L - 1, main + 1)
        base = np.interp(np.arange(L), [lb, rb], [I[lb], I[rb]])
        corr = I - base
        height = float(corr[main])
        area = float(_trapz(np.clip(corr[lb:rb + 1], 0, None), dx=dV))
        Vp = float(V[main])

        pc = ph - ref_ph
        Vp_corr = Vp + mv_per_ph * pc       # remove Nernstian shift -> pH-invariant
        out[i] = [Vp, Vp_corr, pc, pc * height, pc * area, pc * Vp]

    return np.nan_to_num(out).astype(np.float32), list(PH_COUPLED_NAMES)


PH_CURVE_GLOBAL_NAMES = ["auc_abs", "auc_signed", "mean_I", "std_I", "range_I",
                         "centroid_V", "centroid_V_pHcorr", "spread_V",
                         "upper_lower_ratio", "pH_centered",
                         "pHxauc", "pHxmean", "pHxcentroid", "pHxrange"]


def ph_curve_features(current_data, V, ph_data, bin_size: int = 3,
                      ref_ph: float = 7.0, mv_per_ph: float = 0.059):
    """
    PEAK-FREE pH-coupled features. No peak detection -- every descriptor is
    defined for every trace, including flat near-LOD traces.

    Two encodings are returned:

    (A) GLOBAL descriptors of the whole I(V) curve + their pH interactions:
        bulk area, mean/std/range, the current CENTROID (area-weighted mean
        voltage -- a peak-free "where the signal sits"), its Nernst-shifted
        version, curve spread, upper/lower charge ratio, and pH x descriptor
        interactions.

    (B) pH x WHOLE-CURVE: centered-pH multiplied by each binned current value,
        so the model sees pH-modulated current at every voltage with no peak and
        no assumed shift.

    Returns (X_global, global_names, X_phcurve, phcurve_names).
    """
    I = np.asarray(current_data, dtype=np.float64)
    n, L = I.shape
    dV = abs(V[1] - V[0])
    mid = L // 2
    _tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    # baseline-corrected current (linear endpoint baseline) for shape descriptors
    base = np.linspace(I[:, :1], I[:, -1:], L, axis=1).reshape(n, L)
    c = I - base
    w = np.abs(c)
    wsum = w.sum(axis=1) + 1e-12

    auc_abs = _tz(np.abs(c), dx=dV, axis=1)
    auc_signed = _tz(c, dx=dV, axis=1)
    mean_I = I.mean(axis=1)
    std_I = I.std(axis=1)
    range_I = I.max(axis=1) - I.min(axis=1)
    centroid_V = (w * V[None, :]).sum(axis=1) / wsum
    spread_V = np.sqrt((w * (V[None, :] - centroid_V[:, None]) ** 2).sum(axis=1) / wsum)
    upper = w[:, mid:].sum(axis=1); lower = w[:, :mid].sum(axis=1)
    upper_lower_ratio = upper / (lower + 1e-12)

    pc = (np.asarray(ph_data, dtype=np.float64) - ref_ph)
    centroid_V_pHcorr = centroid_V + mv_per_ph * pc      # Nernst-shifted centroid

    Xg = np.column_stack([
        auc_abs, auc_signed, mean_I, std_I, range_I,
        centroid_V, centroid_V_pHcorr, spread_V, upper_lower_ratio, pc,
        pc * auc_abs, pc * mean_I, pc * centroid_V, pc * range_I,
    ]).astype(np.float32)

    # (B) pH x whole binned curve
    n_bins = L // bin_size
    trim = n_bins * bin_size
    I_binned = I[:, :trim].reshape(n, n_bins, bin_size).mean(axis=2)
    Xc = (pc[:, None] * I_binned).astype(np.float32)
    phcurve_names = [f"pHxI_bin{i}" for i in range(n_bins)]

    return np.nan_to_num(Xg), list(PH_CURVE_GLOBAL_NAMES), np.nan_to_num(Xc), phcurve_names


def drop_constant_columns(X, feature_names):
    """
    Remove zero-variance columns (e.g. the analyte/source one-hots inside a
    single-group subset). Returns the reduced matrix, the surviving names, and a
    boolean keep-mask so the same columns can be dropped at inference time.
    """
    X = np.asarray(X)
    keep = X.std(axis=0) > 0
    kept_names = [n for n, k in zip(feature_names, keep) if k]
    return X[:, keep], kept_names, keep


# ── Region-of-interest (ROI) features ────────────────────────────────────────
def define_roi_windows(traces_high, traces_low, V, max_windows: int = 3,
                       uric_cutoff=None, smooth: int = 7, min_prom_frac: float = 0.05,
                       half_width: float = 0.05):
    """
    Locate analyte signal windows from high-concentration traces, excluding the
    uric-acid interferent peak (rightmost, ~0.4-0.5 V).

    Strategy: uric acid is ~constant with analyte concentration, so the
    high-minus-low concentration DIFFERENCE cancels it and lights up the
    analyte-responsive regions. We detect the rightmost prominent peak in the
    high-conc average (uric acid), exclude everything from its left foot
    rightward, find up to `max_windows` peaks in the difference to its left, and
    place a FIXED-WIDTH window (+/- half_width volts) around each peak center so
    windows stay tight on real signal instead of sprawling along baseline.

    Returns (windows, uric_V) where windows is a list of (v_lo, v_hi).
    """
    mh = np.asarray(traces_high, float).mean(axis=0)
    ml = np.asarray(traces_low, float).mean(axis=0)
    L = len(V)
    win = smooth if smooth % 2 == 1 else smooth + 1
    mhs = savgol_filter(mh, win, 2) if L >= win + 2 else mh
    diff = savgol_filter(mh - ml, win, 2) if L >= win + 2 else (mh - ml)

    # uric-acid cutoff: rightmost prominent peak in the high-conc average.
    # Cut just LEFT of the uric peak position (peak minus a margin), NOT at its
    # left_base -- the left_base can run far left past smaller analyte peaks.
    uric_margin = 0.06   # volts to the left of the uric peak center to start excluding
    if uric_cutoff is not None:
        cut_idx = int(np.searchsorted(V, uric_cutoff))
        uric_V = float(uric_cutoff)
    else:
        pks, props = find_peaks(mhs, prominence=(np.ptp(mhs) + 1e-12) * min_prom_frac)
        if len(pks):
            r = int(np.argmax(V[pks]))           # rightmost peak = uric acid
            uric_V = float(V[pks[r]])
            cut_idx = int(np.searchsorted(V, uric_V - uric_margin))
        else:
            cut_idx, uric_V = L, None
    cut_idx = max(5, min(cut_idx, L))

    # analyte peaks in the difference: find on the CLEAN difference, then filter
    # by position (left of uric, off the edges) and sign (positive = analyte).
    edge = max(2, int(0.04 * L))
    dpks, dprops = find_peaks(diff, prominence=(np.ptp(diff) + 1e-12) * min_prom_frac)
    if len(dpks):
        proms = dprops["prominences"]
        valid = [(p, pr) for p, pr in zip(dpks, proms)
                 if edge <= p < cut_idx and diff[p] > 0]
        valid.sort(key=lambda t: t[1], reverse=True)          # by prominence
        chosen = sorted([p for p, _ in valid[:max_windows]])  # then left->right
    else:
        chosen = []

    def _window_around(center_idx):
        vc = float(V[center_idx])
        lo = max(vc - half_width, float(V[0]))
        hi = min(vc + half_width, float(V[cut_idx - 1]))      # never cross uric
        return (lo, hi)

    windows = [_window_around(p) for p in chosen]
    if not windows:                               # fallback: tight window at max positive diff
        masked = np.where((np.arange(L) >= edge) & (np.arange(L) < cut_idx), diff, -np.inf)
        windows = [_window_around(int(np.argmax(masked)))]
    return windows, uric_V


def roi_features(current_data, V, windows):
    """Integrated current, mean, and max within each ROI window, per trace.
    Defined for every trace (incl. peakless low-concentration ones)."""
    I = np.asarray(current_data, float)
    n, L = I.shape
    dV = abs(V[1] - V[0])
    cols, names = [], []
    for wi, (vlo, vhi) in enumerate(windows):
        i0 = int(np.searchsorted(V, vlo)); i1 = int(np.searchsorted(V, vhi))
        i0, i1 = max(0, min(i0, L - 2)), max(1, min(i1, L - 1))
        if i1 <= i0:
            i1 = i0 + 1
        seg = I[:, i0:i1 + 1]
        auc = _trapz(seg, dx=dV, axis=1)
        cols += [auc, seg.mean(axis=1), seg.max(axis=1)]
        names += [f"roi{wi}_auc", f"roi{wi}_mean", f"roi{wi}_max"]
    X = np.column_stack(cols).astype(np.float32)
    return np.nan_to_num(X), names