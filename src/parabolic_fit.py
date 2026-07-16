"""
parabolic_fit.py
================
Constrained peak fitting, ported from Peakfit_DPV.ipynb / Peakfit_SWV.ipynb.

Model (fit to the RAW trace -- the parabola absorbs the background):

    y(x) = bas1 + mag1*(x - off1)^2                       <- parabolic baseline
         + sum_k  amp_k/(sig_k*sqrt(2pi)) * exp(-0.5*((x-mu_k)/sig_k)^2)

Eight Gaussians are fitted with their centers CONSTRAINED to known
electrochemical positions (different for DPV and SWV) with a small tolerance.
This is the key property: because the centers are priors rather than something
we must detect, a peak can still be fitted when it is small or hidden -- its
amplitude simply comes out near zero. Auto-detection cannot do this, which is why
our earlier peak features failed at low concentration.

Features returned per trace = the 24 Gaussian parameters (amp, center, width for
each of the 8 peaks), matching Process_ML.ipynb which drops the 3 parabola params.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# fixed peak centers (volts) from the notebooks
CENS_DPV = [-0.18, -0.11, -0.02, -0.3, 0.09, 0.26, 0.37, 0.43]
CENS_SWV = [-0.1, -0.2, -0.33, 0.0, 0.13, 0.24, 0.37, 0.45]

# initial guesses (from the notebooks)
BAS1, MAG1, OFF1 = 30.0, 20.0, 0.1
AMP0, SIGMA0 = 10.0, 0.05


def _model(x, *p):
    """parabola + N gaussians; p = [bas1, mag1, off1, (amp, mu, sig) x N]"""
    bas1, mag1, off1 = p[0], p[1], p[2]
    out = bas1 + mag1 * (x - off1) ** 2
    rest = p[3:]
    for i in range(0, len(rest), 3):
        amp, mu, sig = rest[i], rest[i + 1], rest[i + 2]
        out = out + amp * (1.0 / (sig * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return out


def _setup(technique: str):
    """Return (p0, lower, upper) with the notebook's per-technique bounds."""
    tech = (technique or "DPV").upper()
    if tech == "SWV":
        cens = CENS_SWV
        # SWV notebook: all peaks use +/-0.02 center tolerance, sigma in [0.01, 0.07]
        specs = [(c, 0.02, 0.01, 0.07) for c in cens]
    else:
        cens = CENS_DPV
        # DPV notebook: first three use +/-0.03, sigma [0.001, 0.05];
        #               remaining use +/-0.03, sigma [0.01, 0.07]
        specs = [(c, 0.03, 0.001, 0.05) for c in cens[:3]] + \
                [(c, 0.03, 0.01, 0.07) for c in cens[3:]]

    p0 = [BAS1, MAG1, OFF1]
    lo = [-np.inf, -np.inf, -np.inf]
    hi = [np.inf, np.inf, np.inf]
    for c, ctol, smin, smax in specs:
        p0 += [AMP0, c, SIGMA0]
        lo += [0.0, c - ctol, smin]
        hi += [np.inf, c + ctol, smax]
    return np.array(p0, float), np.array(lo, float), np.array(hi, float), cens


def feature_names(technique: str = "DPV"):
    _, _, _, cens = _setup(technique)
    names = []
    for k, c in enumerate(cens):
        names += [f"g{k}_amp@{c:+.2f}", f"g{k}_ctr@{c:+.2f}", f"g{k}_wid@{c:+.2f}"]
    return names


def fit_trace(V, I, technique: str = "DPV", maxfev: int = 10000):
    """Fit one trace. Returns (gauss_params_24, ok_flag, resid_rms).
    On failure returns zeros with ok=False (mirrors the notebook's FAILED path)."""
    p0, lo, hi, _ = _setup(technique)
    n_g = len(p0) - 3
    try:
        popt, _ = curve_fit(_model, V, I, p0=p0, bounds=(lo, hi), maxfev=maxfev)
        resid = I - _model(V, *popt)
        return popt[3:].astype(np.float32), True, float(np.sqrt(np.mean(resid ** 2)))
    except Exception:
        return np.zeros(n_g, np.float32), False, float("nan")


def fit_all(traces, V, techniques, verbose_every: int = 500):
    """Fit every trace. Returns (X_gauss [n,24], ok_mask, resid_rms).
    `techniques` is a per-trace sequence of 'DPV'/'SWV' so the right center
    priors and bounds are used for each."""
    traces = np.asarray(traces, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(traces)
    n_feat = 24
    X = np.zeros((n, n_feat), np.float32)
    ok = np.zeros(n, bool)
    rms = np.full(n, np.nan, float)
    for i in range(n):
        L = min(len(V), traces.shape[1])
        p, good, r = fit_trace(V[:L], traces[i, :L], str(techniques[i]))
        if len(p) == n_feat:
            X[i] = p
        ok[i] = good
        rms[i] = r
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    fitted {i+1}/{n}  ({ok[:i+1].mean()*100:.0f}% ok)")
    return X, ok, rms