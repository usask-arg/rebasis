---
file_format: mystnb
kernelspec:
  name: python3
---

# Regridding

{func}`~rebasis.regrid` maps coefficients in one space to another along an axis of an array;
{func}`~rebasis.regrid_nd` does several axes, and {func}`rebasis.xarray.regrid` works on
xarray objects by dimension name. Methods:

- `"project"` (default): the $L^2$ projection, conservative.
- `"measure"`: the target elements are averaging kernels.
- `"interpolate"`: point values at the target centers.
- `"fit"`: the source elements are the kernels that measured the data (e.g. bin averages),
  the target a model space: smooth conservative upsampling.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import rebasis as rb
from _plot import BLUE, MUTED, ORANGE
```

## Bounded projection

$L^2$ projection onto a coarse smooth basis rings near discontinuities. `bounded=True` finds the
best approximation that conserves the integral and creates no new extrema (each coefficient
within the range of the inputs it overlaps); `bounded="positive"` and `bounded=(lo, hi)` apply
global bounds.

```{code-cell} ipython3
x = np.linspace(0, 1, 4001)
f = ((x > 0.3) & (x < 0.55)).astype(float)
src = rb.Space.hats(x)
dst = rb.Space.bsplines_clamped(np.linspace(0, 1, 16), 3)
plain = rb.regrid(f, src, dst)
bounded = rb.regrid(f, src, dst, bounded=True)

xx = np.linspace(0, 0.9999, 1000)
E = dst.evaluate(xx)
m = dst.integrals()
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(x, f, color=MUTED, lw=1, label=f"input (∫ = {f @ src.integrals():.6f})")
ax.plot(xx, E @ plain, color=BLUE, label=f"L2 projection (∫ = {plain @ m:.6f})")
ax.plot(xx, E @ bounded, color=ORANGE, label=f"bounded (∫ = {bounded @ m:.6f})")
ax.legend(loc="upper right");
```

## Smooth conservative upsampling

From bin averages to a spline whose bin averages reproduce them:

```{code-cell} ipython3
edges = np.linspace(0, 2 * np.pi, 9)
avg = np.diff(-np.cos(edges)) / np.diff(edges)  # bin averages of sin
bins = rb.Space.boxes(edges)
model = rb.Space.bsplines_clamped(edges, 3)
coef = rb.regrid(avg, bins, model, method="fit", smoothing=1e-8)

xx = np.linspace(0, 2 * np.pi * 0.9999, 500)
fig, ax = plt.subplots(figsize=(7, 3))
ax.stairs(avg, edges, color=MUTED, label="bin averages")
ax.plot(xx, model.evaluate(xx) @ coef, color=BLUE, label="spline with the same bin averages")
ax.legend(loc="lower left")
print("bin averages reproduced to", np.abs(rb.measure(bins, model) @ coef - avg).max())
```

## Area-weighted latitude–longitude regridding

Grids that are tensor products are regridded one axis at a time, which is exact for separable
weights. `"cos_lat_deg"` weights latitude by area:

```{code-cell} ipython3
lat_f, lon_f = rb.Space.boxes(np.linspace(-90, 90, 181)), rb.Space.boxes(np.linspace(0, 360, 361))
lat_c, lon_c = rb.Space.boxes(np.linspace(-90, 90, 13)), rb.Space.boxes(np.linspace(0, 360, 25))
lat, lon = np.meshgrid(lat_f.centers, lon_f.centers, indexing="ij")
field = np.cos(np.radians(lat)) ** 2 * (1 + 0.3 * np.sin(np.radians(3 * lon)))

coarse = rb.regrid_nd(field, {0: (lat_f, lat_c, "cos_lat_deg"), 1: (lon_f, lon_c)})


def global_mean(v, lat_s, lon_s):
    w = np.outer(lat_s.integrals("cos_lat_deg"), lon_s.integrals())
    return (v * w).sum() / w.sum()


print(global_mean(field, lat_f, lon_f), global_mean(coarse, lat_c, lon_c))
```

## xarray

```{code-cell} ipython3
import xarray as xr

import rebasis.xarray

da = xr.DataArray(field, dims=("lat", "lon"), coords={"lat": lat_f.centers, "lon": lon_f.centers})
rebasis.xarray.regrid(
    da,
    {"lat": np.arange(-87.5, 90, 5.0), "lon": np.arange(2.5, 360, 5.0)},
    weight={"lat": "cos_lat_deg"},
)
```

Missing data: `skipna=True` regrids with NaNs set to zero and divides by the regridded validity
mask, so outputs are averages of the valid inputs (`na_thres` sets the minimum valid fraction).
