"""Summary figure of the sampling study and the real-OSIRIS estimates (from out/*.npz)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, INK_2, MUTED, ORANGE, apply

apply()
out = Path(__file__).with_name("out")
cases = [
    ("Sep 2010, 46 hPa", "sampling_global_201009_46hPa.npz"),
    ("Sep 2010, 10 hPa", "sampling_global_201009_10hPa.npz"),
    ("Mar 2010, 46 hPa", "sampling_global_201003_46hPa.npz"),
    ("Nov 2010, 46 hPa", "sampling_global_201011_46hPa.npz"),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (title, name) in zip(axes.flat[:4], cases, strict=True):
    d = np.load(out / name)
    c = 0.5 * (d["edges"][1:] + d["edges"][:-1])
    rms = lambda e: np.sqrt(np.nanmean(e**2))  # noqa: E731
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(
        c,
        d["bin"] - d["truth"],
        "o",
        color=ORANGE,
        ms=5,
        label=f"bin average (RMS {rms(d['bin'] - d['truth']):.3f})",
    )
    ax.errorbar(
        c,
        d["est"] - d["truth"],
        yerr=2 * d["std"],
        fmt="o",
        color=BLUE,
        ms=5,
        capsize=2,
        label=f"optimal estimate ± 2σ (RMS {rms(d['est'] - d['truth']):.3f})",
    )
    ax.set_title(f"sampling error, {title}", loc="left")
    ax.set_ylabel("error [ppmv]")
    ax.legend(loc="lower left", fontsize=7)
for ax, alt in zip(axes.flat[4:], [20, 30], strict=True):
    d = np.load(out / f"osiris_201009_{alt}km.npz")
    c = 0.5 * (d["edges"][1:] + d["edges"][:-1])
    ax.axhline(0, color=MUTED, lw=1)
    ax.errorbar(
        c,
        d["est"] - d["bin"],
        yerr=2 * d["std"],
        fmt="o",
        color=BLUE,
        ms=5,
        capsize=2,
        label="estimate − bin average (± 2σ of estimate)",
    )
    ax.errorbar(
        c,
        np.zeros_like(c),
        yerr=2 * d["sem"],
        fmt="none",
        color=ORANGE,
        capsize=2,
        label="± 2 × standard error of bin mean",
    )
    ax.set_title(f"real OSIRIS, Sep 2010, {alt} km", loc="left")
    ax.set_ylabel("[10¹² cm⁻³]")
    ax.legend(loc="lower right", fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel("latitude band center [°]")
axes[0, 0].set_xlabel("latitude band center [°]")
fig.suptitle(
    "OSIRIS monthly zonal means: bin averages vs rebasis optimal estimates "
    "(sampling error = OSIRIS-sampled MLS field minus its true zonal mean)",
    x=0.01,
    ha="left",
    fontsize=11,
    color=INK_2,
)
fig.tight_layout()
path = out / "osiris_summary.png"
fig.savefig(path, dpi=120)
print(f"wrote {path}")
