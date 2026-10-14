"""Timings at line-by-line scale: build the instrument matrix, apply it, and regrid."""

import time

import numpy as np

import rebasis as rb


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"{label:<55s} {time.perf_counter() - t0:8.3f} s")
    return out


x = np.linspace(500.0, 1500.0, 1_000_001)  # 0.001 nm sampling
hires = timed("hats space, 1e6 nodes", lambda: rb.Space.hats(x))
centers = np.linspace(510, 1490, 1000)

for name, ils in [
    ("gaussian ILS (fwhm 0.5 nm)", rb.Space.gaussians(centers, fwhm=0.5)),
    (
        "tabulated ILS (101-pt triangle, 1 nm wide)",
        rb.Space.shifted(centers, np.linspace(-0.5, 0.5, 101), 1 - np.abs(np.linspace(-1, 1, 101))),
    ),
]:
    w = timed(f"measure: 1000 x {name}", lambda ils=ils: rb.measure(ils, hires))
    print(f"    nnz = {w.nnz:,} ({w.data.nbytes / 1e6:.0f} MB data; dense would be 8 GB)")
    f = np.random.default_rng(0).random((20, x.size))  # e.g. 20 lines of sight
    timed("    apply to (20, 1e6) radiances", lambda w=w, f=f: f @ w.T)

coarse = rb.Space.boxes(np.linspace(500, 1500, 10_001))
p = timed("projection setup, 1e6 hats -> 1e4 boxes", lambda: rb.project(hires, coarse))
timed("    apply to (20, 1e6)", lambda: p.apply(f))
coarse_hats = rb.Space.hats(np.linspace(500, 1500, 10_001))
p = timed("projection setup, 1e6 hats -> 1e4 hats", lambda: rb.project(hires, coarse_hats))
timed("    apply to (20, 1e6)", lambda: p.apply(f))
