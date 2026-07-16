"""Level-averaged parity plot with dual SD + SE error bars (shared)."""
from __future__ import annotations
import numpy as np

C_POINT = "#16202B"
C_SD = "#9FC0D2"
C_SE = "#A32D2D"
C_RAW = "#CCCCCC"
C_IDEAL = "#2C2C2A"


def calibration_plot(yt, yp, label, out, fmt_conc=None, show_raw=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    ok = ~np.isnan(yp)
    yt, yp = yt[ok], yp[ok]

    ytr = np.round(yt, 3)
    levels = sorted(np.unique(ytr))

    means, sds, ses, maes, ns = [], [], [], [], []
    for lv in levels:
        v = yp[ytr == lv]
        n = max(1, len(v))
        sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
        means.append(float(np.mean(v)))
        sds.append(sd)
        ses.append(sd / np.sqrt(n) if n > 1 else 0.0)
        maes.append(float(np.mean(np.abs(v - lv))))
        ns.append(n)
    means = np.array(means); sds = np.array(sds); ses = np.array(ses)

    lims = [min(min(levels), yp.min()) - 0.3, max(max(levels), yp.max()) + 0.3]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    if show_raw:
        ax.scatter(yt, yp, color=C_RAW, alpha=0.45, s=14, zorder=1,
                   label="individual predictions")
    ax.errorbar(levels, means, yerr=sds, fmt="none", ecolor=C_SD,
                elinewidth=5, capsize=0, alpha=0.9, zorder=2, label="\u00b1 SD (spread)")
    ax.errorbar(levels, means, yerr=ses, fmt="none", ecolor=C_SE,
                elinewidth=2, capsize=5, zorder=3, label="\u00b1 SE (of mean)")
    ax.plot(levels, means, "o", color=C_POINT, ms=9, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5, label="mean prediction")
    ax.plot(lims, lims, "--", color=C_IDEAL, lw=1.3, zorder=1.5, label="ideal")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("True log10(M)")
    ax.set_ylabel("Predicted log10(M)")
    omae = float(np.mean(np.abs(yp - yt)))
    ax.set_title(f"{label}\nmean \u00b1 SE (red) and \u00b1 SD (blue)  |  "
                 f"overall MAE {omae:.2f} ({10**omae:.1f}x)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close(fig)

    print(f"  [{label}] level-averaged:")
    for lv, mu, sd, se, ma, n in zip(levels, means, sds, ses, maes, ns):
        name = fmt_conc(lv) if fmt_conc else f"{lv:.2f}"
        print(f"    {name:<12} n={n:>3}  mean={mu:+.2f}  SD={sd:.3f}  SE={se:.3f}"
              f"   [per-sample MAE {ma:.2f} = {10**ma:.1f}x]")
    return None
