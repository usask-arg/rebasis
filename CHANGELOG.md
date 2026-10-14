# Changelog

## 0.1.0

First release.

- Spaces of boxes, hats, B-splines (degree 0-3), deltas, and line-shape profiles: Gaussian,
  Lorentzian, Voigt, pseudo-Voigt, super-Gaussian (optionally asymmetric), smoothed box, FTS line
  shapes for cosine-sum and Norton–Beer apodizations, rectangles, triangles, trapezoids, and
  tabulated shapes (linear or PCHIP, shared or per channel).
- `measure`, `evaluate`, `Projection` (optionally bounded: local, positive or global bounds,
  conservative), `Fit` (penalized least squares: smoothing splines, histopolation).
- `measure_derivative`: exact derivatives with respect to center, scale and profile parameters.
- Elements across coordinates (`to_wavenumber`, `to_wavelength`, `reciprocal`, `affine`).
- `regrid`, `regrid_nd` and `rebasis.xarray.regrid`, with area weighting and missing data.
