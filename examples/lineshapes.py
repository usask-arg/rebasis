"""Gallery of every line shape, each normalized to unit area. Writes examples/lineshapes.png."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import BLUE, INK_2, apply

import rebasis as rb

apply()
c = [0.0]
shapes = [
    ("Gaussian, FWHM 1", rb.Space.gaussians(c, fwhm=1.0)),
    ("Lorentzian, FWHM 1", rb.Space.lorentzians(c, fwhm=1.0)),
    ("Voigt, 0.8 G + 0.4 L", rb.Space.voigts(c, fwhm_gaussian=0.8, fwhm_lorentzian=0.4)),
    ("pseudo-Voigt, η = 0.5", rb.Space.pseudo_voigts(c, fwhm=1.0, eta=0.5)),
    ("super-Gaussian, p = 4", rb.Space.super_gaussians(c, fwhm=1.0, power=4)),
    (
        "super-Gaussian, p = 3, asym 0.3",
        rb.Space.super_gaussians(c, fwhm=1.0, power=3, asymmetry=0.3),
    ),
    ("smoothed box, 1 ⊗ G(0.25)", rb.Space.smoothed_boxes(c, width=1.0, fwhm=0.25)),
    ("FTS, no apodization (sinc)", rb.Space.fts(c, max_opd=0.6, apodization="none")),
    ("FTS, triangle (sinc²)", rb.Space.fts(c, max_opd=0.6, apodization="triangle")),
    ("FTS, Hamming", rb.Space.fts(c, max_opd=0.6, apodization="hamming")),
    ("FTS, Norton–Beer strong", rb.Space.fts(c, max_opd=0.6, apodization="norton_beer_strong")),
    ("FTS, Blackman–Harris", rb.Space.fts(c, max_opd=0.6, apodization="blackman_harris")),
    ("rectangle", rb.Space.rectangles(c, width=1.0)),
    ("triangle, FWHM 1", rb.Space.triangles(c, fwhm=1.0)),
    ("trapezoid 0.6 / 1.4", rb.Space.trapezoids(c, top_width=0.6, bottom_width=1.4)),
    (
        "tabulated (PCHIP)",
        rb.Space.shifted(
            c, [-1.2, -0.6, -0.2, 0.0, 0.5, 1.0], [0, 0.25, 0.9, 1.0, 0.3, 0], interp="pchip"
        ),
    ),
]

x = np.linspace(-3, 3, 1201)
fig, axes = plt.subplots(4, 4, figsize=(11, 8.5), sharex=True)
for ax, (title, s) in zip(axes.flat, shapes, strict=True):
    y = s.evaluate(x).toarray()[:, 0] / s.integrals()[0]  # unit area
    ax.plot(x, y, color=BLUE)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.set_title(title, loc="left")
for ax in axes[-1]:
    ax.set_xlabel("offset from line center")
fig.suptitle("rebasis line shapes (unit area)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
out = Path(__file__).with_name("lineshapes.png")
fig.savefig(out, dpi=120)
print(f"wrote {out}")
