"""Bounded projection and conservative upsampling. Writes examples/regridding.png."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import BLUE, INK_2, MUTED, ORANGE, apply

import rebasis as rb

apply()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# 1. a step projected onto coarse cubic B-splines, with and without bounds
x = np.linspace(0, 1, 4001)
f = ((x > 0.3) & (x < 0.55)).astype(float)
src = rb.Space.hats(x)
dst = rb.Space.bsplines_clamped(np.linspace(0, 1, 16), 3)
plain = rb.regrid(f, src, dst)
bounded = rb.regrid(f, src, dst, bounded=True)
xx = np.linspace(0, 0.9999, 1000)
E = dst.evaluate(xx)
m = dst.integrals()
ax1.plot(x, f, color=MUTED, lw=1, label=f"input step (∫ = {f @ src.integrals():.6f})")
ax1.plot(xx, E @ plain, color=BLUE, label=f"L2 projection (∫ = {plain @ m:.6f})")
ax1.plot(xx, E @ bounded, color=ORANGE, label=f"bounded projection (∫ = {bounded @ m:.6f})")
ax1.legend(loc="upper right", labelcolor=INK_2)
ax1.set_title("A step projected onto 15-interval cubic B-splines", loc="left")
ax1.set_xlabel("x")

# 2. conservative smooth upsampling of bin averages
edges = np.linspace(0, 2 * np.pi, 9)
avg = np.diff(-np.cos(edges)) / np.diff(edges)
bins = rb.Space.boxes(edges)
model = rb.Space.bsplines_clamped(edges, 3)
coef = rb.regrid(avg, bins, model, method="fit", smoothing=1e-8)
xx = np.linspace(0, 2 * np.pi * 0.9999, 1000)
ax2.stairs(avg, edges, color=MUTED, lw=1, label="8 bin averages of sin(x)")
ax2.plot(xx, np.sin(xx), color=MUTED, lw=1, ls="--", label="sin(x)")
ax2.plot(xx, model.evaluate(xx) @ coef, color=BLUE, label="cubic spline with the same bin averages")
ax2.legend(loc="lower left", labelcolor=INK_2)
ax2.set_title("Conservative smooth upsampling (method='fit')", loc="left")
ax2.set_xlabel("x")

fig.tight_layout()
out = Path(__file__).with_name("regridding.png")
fig.savefig(out, dpi=120)
print(f"wrote {out}")
