# rebasis

Linear maps between function bases, with a Rust core and a Python interface.

A quantity is represented as f(x) = Σ_j f_j B_j(x). rebasis computes the sparse matrices that
turn the coefficients f_j into:

| operation | what it computes | typical use |
|---|---|---|
| `measure(kernels, basis)` | g_i = ∫ K_i f / ∫ K_i | instrument line shape applied to a high-resolution spectrum |
| `evaluate(basis, x)` | g_i = f(x_i) | interpolation, plotting a reconstruction |
| `project(src, dst)` | best L2 fit Σ g_i G_i ≈ f | regridding (conservative), up- or down-sampling |

Inner products are exact for piecewise polynomials (boxes, hats, tabulated line shapes) and
accurate to rounding for Gaussians; there are no tolerance or resolution parameters.

```python
import numpy as np
import rebasis as rb

wavel = np.linspace(300, 310, 100_001)         # high-resolution model grid
hires = rb.Space.hats(wavel)                   # radiance = linear interpolant of the samples
ils = rb.Space.gaussians(np.arange(301, 309, 0.1), fwhm=0.5)

W = rb.measure(ils, hires)                     # scipy.sparse.csr_array, (80, 100001)
measured = W @ radiance                        # or rb.apply_along(W, radiance, axis=...)

coarse = rb.Space.boxes(np.arange(300, 310.01, 0.5))
binned = rb.project(hires, coarse).apply(radiance)   # conserves the integral
```

## Elements and conventions

Elements are shapes, not densities: boxes, hats and Gaussians peak at 1. Normalization is a
property of the operation (`measure(..., normalization="integral" | "rowsum" | "none")`), not of
the element, so the same space can serve as a basis in one call and as kernels in another.

- `Space.hats(nodes)`: piecewise linear; coefficients are values at the nodes. The right basis
  for a function known by samples.
- `Space.boxes(edges)`, `Space.boxes_from_centers(centers)`: piecewise constant; coefficients are
  bin averages.
- `Space.gaussians(centers, sigma=... | fwhm=..., cutoff=6)`.
- `Space.shifted(centers, shape_x, shape_y, stretch=1)`: a tabulated line shape, stored once.
- `Space.deltas(x)`: point evaluation as kernels; point masses as a basis.

## Development

```bash
uv sync            # builds the extension (release profile)
uv run pytest
cargo test -p rebasis
```

The Rust crate `rebasis` (in `rebasis/`) has no Python dependency and is intended to be used
directly by sasktran2; `rebasis-py/` holds the pyo3 bindings.
