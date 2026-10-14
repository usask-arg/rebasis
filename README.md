# rebasis

Linear maps between function bases, with a Rust core and a Python interface.

A quantity is represented as f(x) = Σ_j f_j B_j(x). rebasis computes the sparse matrices that
turn the coefficients f_j into:

| operation | what it computes | typical use |
|---|---|---|
| `measure(kernels, basis)` | g_i = ∫ K_i f / ∫ K_i | instrument line shape applied to a high-resolution spectrum |
| `evaluate(basis, x)` | g_i = f(x_i) | interpolation, plotting a reconstruction |
| `project(src, dst)` | best L2 fit Σ g_i G_i ≈ f | conservative regridding, up- or down-sampling |
| `Fit(kernels, model)` | model coefficients whose measurements match data | smooth conservative upsampling, smoothing splines, inverting `measure` |

Inner products are exact for piecewise polynomials (boxes, hats, B-splines, tabulated shapes) and
accurate to rounding for the analytic line shapes; there are no tolerance or resolution
parameters.

```python
import numpy as np
import rebasis as rb

wavel = np.linspace(300, 310, 100_001)         # high-resolution model grid
hires = rb.Space.hats(wavel)                   # radiance = linear interpolant of the samples
ils = rb.Space.voigts(np.arange(301, 309, 0.1), fwhm_gaussian=0.4, fwhm_lorentzian=0.05)

W = rb.measure(ils, hires)                     # scipy.sparse.csr_array, (80, 100001)
measured = W @ radiance                        # or rb.apply_along(W, radiance, axis=...)

coarse = rb.Space.boxes(np.arange(300, 310.01, 0.5))
binned = rb.regrid(radiance, hires, coarse)    # conservative; bounded=True to avoid overshoot
```

## Spaces

Elements are shapes, not densities: everything peaks at 1. Normalization is a property of the
operation (`measure(..., normalization="integral" | "rowsum" | "none")`), not of the element, so
the same space can serve as a basis in one call and as kernels in another. Per-element widths
can be scalars or arrays.

**Bases**

- `Space.hats(nodes)`: piecewise linear; coefficients are values at the nodes. The basis for a
  function known by samples.
- `Space.boxes(edges)`, `Space.boxes_from_centers(centers)`: piecewise constant; coefficients
  are bin averages.
- `Space.bsplines_clamped(breakpoints, degree)`, `Space.bsplines(knots, degree)`: B-splines of
  degree 0 to 3.
- `Space.deltas(x)`: point evaluation as kernels, point masses as a basis.

**Line shapes** (`cutoff` is the half width of the truncated support, in the unit given)

| constructor | shape | cutoff unit |
|---|---|---|
| `gaussians(c, sigma= \| fwhm=)` | Gaussian | σ (default 6) |
| `lorentzians(c, hwhm= \| fwhm=)` | Lorentzian | HWHM (100) |
| `voigts(c, sigma= \| fwhm_gaussian=, gamma= \| fwhm_lorentzian=)` | Voigt (Faddeeva, ~3e-15) | Voigt HWHM (100) |
| `pseudo_voigts(c, fwhm=, eta=)` | η·L + (1−η)·G | HWHM (100) |
| `super_gaussians(c, fwhm=, power=, asymmetry=0)` | exp(−ln2·\|2t/w\|^p), optionally asymmetric | HWHM per side (6) |
| `smoothed_boxes(c, width=, sigma= \| fwhm=)` | slit ⊗ Gaussian | σ past the box edge (6) |
| `fts(c, max_opd=, apodization=)` | FTS line shape: `none` (sinc), `triangle` (sinc²), `hann`, `hamming`, `blackman`, `blackman_harris`, `norton_beer_weak/medium/strong`, or custom cosine-sum / Norton–Beer coefficients | 1/(2·max_opd) (100) |
| `rectangles(c, width=)`, `triangles(c, fwhm=)`, `trapezoids(c, top_width=, bottom_width=)` | exact piecewise linear | |
| `shifted(c, shape_x, shape_y, stretch=1, interp="linear" \| "pchip")` | one tabulated shape, stored once | |
| `tabulated(c, x, y, interp=)` | a tabulated shape per channel | |

`examples/lineshapes.py` plots all of them.

## Regridding

- `rb.regrid(data, src, dst, axis, method=...)`: `"project"` (conservative, default),
  `"measure"` (`dst` as averaging kernels), `"interpolate"`, or `"fit"` (smooth conservative
  upsampling of bin averages onto a spline `dst`).
  - `bounded=True` gives the best L2 approximation that conserves the integral and creates no new
    extrema: each coefficient stays within the range of the inputs it overlaps. It solves a
    small banded quadratic program exactly, so corrections stay local.
    `bounded="positive"` and `bounded=(lo, hi)` apply global bounds instead.
  - `skipna=True` handles missing data.
  - `weight="cos_lat_deg"` gives area weighting in latitude.
- `rb.regrid_nd(data, {axis: (src, dst[, weight]), ...})`: tensor-product grids, one axis at a
  time (exact for separable grids).
- `rebasis.xarray.regrid(ds, {"lat": new_lat, "lon": new_lon}, weight={"lat": "cos_lat_deg"})`
  regrids DataArrays and Datasets by dimension name.
- `rb.Fit(kernels, model, smoothing=..., penalty="derivative", order=2)` is the general
  penalized least-squares fit. The penalty is ∫(f⁽ʳ⁾)² (a smoothing spline) or coefficient
  differences (P-splines).

## Derivatives with respect to line-shape parameters

`rb.measure_derivative(kernels, basis, parameter)` returns `(W, dW)`, where `dW` is ∂W/∂p with
row i differentiated with respect to kernel i's parameter. For a spectrum `f`, `dW @ f` is the
Jacobian of the measurements with respect to per-channel parameters (diagonal), or with respect
to one parameter shared by all channels. The derivatives are exact: they include the moving
truncation edges and the normalization, and they agree with finite differences to the accuracy
of the finite differences.

| parameter | kernels |
|---|---|
| `center` | all (spectral shift) |
| `scale` | all but deltas: width · ∂/∂width, every width scaled together (squeeze or stretch) |
| `sigma` | Gaussian, Voigt, smoothed box |
| `gamma` | Lorentzian, Voigt |
| `eta` | pseudo-Voigt |
| `power`, `asymmetry` | super-Gaussian |
| `half_width` | smoothed box |

## Development

```bash
uv sync            # builds the extension (release profile)
uv run pytest
cargo test -p rebasis
```

The Rust crate `rebasis` (in `rebasis/`) has no Python dependency and is intended to be used
directly by sasktran2; `rebasis-py/` holds the pyo3 bindings. `examples/` has a benchmark and
figure scripts.
