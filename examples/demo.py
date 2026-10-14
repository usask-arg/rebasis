"""Instrument line shape, regridding and reconstruction on a synthetic spectrum.

Writes examples/demo.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rebasis as rb

rng = np.random.default_rng(3)
x = np.linspace(760.0, 770.0, 20001)  # 0.0005 nm model grid
lines = rng.uniform(760.5, 769.5, 60)
depth = rng.uniform(0.2, 0.9, 60)
tau = sum(d * np.exp(-0.5 * ((x - c) / 0.004) ** 2) for c, d in zip(lines, depth, strict=True))
f = np.exp(-2.0 * tau)
hires = rb.Space.hats(x)

# 1. instrument: Gaussian line shape sampled every 0.1 nm, and a tabulated asymmetric one
centers = np.arange(761.0, 769.01, 0.1)
gauss = rb.measure(rb.Space.gaussians(centers, fwhm=0.4), hires) @ f
sx = np.linspace(-0.4, 0.6, 51)
sy = np.where(sx < 0, 1 + sx / 0.4, 1 - sx / 0.6) ** 2  # skewed line shape
tab = rb.measure(rb.Space.shifted(centers, sx, sy), hires) @ f

# 2. regridding: conservative projection onto 0.25 nm boxes and onto 0.25 nm hats
edges = np.arange(760.0, 770.01, 0.25)
boxes = rb.Space.boxes(edges)
g_box = rb.project(hires, boxes).apply(f)
coarse_hats = rb.Space.hats(edges)
g_hat = rb.project(hires, coarse_hats).apply(f)

# reconstruct the coarse representations on the fine grid for plotting
xx = np.linspace(760.0, 770.0, 4001)[:-1]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(x, f, lw=0.5, color="0.6", label="model (hats, 0.0005 nm)")
ax1.plot(centers, gauss, ".-", label="Gaussian ILS, FWHM 0.4 nm")
ax1.plot(centers, tab, ".-", label="tabulated skewed ILS")
ax1.set_ylabel("transmittance")
ax1.legend(loc="lower left", fontsize=8)
ax1.set_title("measure: instrument line shapes")

ax2.plot(x, f, lw=0.5, color="0.6", label="model")
ax2.plot(xx, boxes.evaluate(xx) @ g_box, label="projection onto 0.25 nm boxes")
ax2.plot(xx, coarse_hats.evaluate(xx) @ g_hat, label="projection onto 0.25 nm hats")
ax2.set_xlabel("wavelength [nm]")
ax2.set_ylabel("transmittance")
ax2.legend(loc="lower left", fontsize=8)
total = f @ hires.integrals()
ax2.set_title(
    "project: conservative regridding  "
    f"(∫f: model {total:.6f}, boxes {g_box @ boxes.integrals():.6f}, "
    f"hats {g_hat @ coarse_hats.integrals():.6f})"
)
fig.tight_layout()
out = Path(__file__).with_name("demo.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
