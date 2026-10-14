# Changelog

## Unreleased

- `Csr::apply_axis` and `Csr::apply_transpose_axis`: apply a matrix or its transpose along any
  axis of an n-dimensional `ndarray` array, e.g. the wavelength axis of sasktran2 radiances and
  their derivatives. `apply` and `apply_last_axis` no longer copy contiguous inputs.
- `Space.concatenate` (`Space::concatenate` in Rust): one space from kernels of different
  shapes, e.g. a different line shape per channel.

## 0.2.0

Estimation: optimal (kriging) estimates of averages from sparse, irregular samples, with
covariance fitting, and an xarray front-end.

- `optimal_estimate`: kriging estimates of grid averages from irregular observations, with
  exact covariances between points, footprints and area-weighted boxes (squared-exponential and
  Matérn ½, 3/2, 5/2 families, periodic dimensions), constant or basis-function means, posterior
  uncertainties, weights and the (REML) log likelihood; `log_likelihood` for fitting.
- `Space.take`; Matérn profiles in the Rust crate.
- Estimation: `values` with k columns (shared covariance; one factorization per group of
  columns with the same valid observations), `MeanTerm` for mean terms with Gaussian priors,
  and `CovarianceSum` (`Covariance + Covariance`) for multi-scale priors.
- Estimation: `neighborhood=` estimates blocks of targets from the observations near them (with
  periodic wrap), for data sets far beyond a single dense solve; `Estimate.n_used`.
- `fit_covariance`: maximum (restricted) likelihood for variances, correlation lengths, a nugget
  and mean-term prior widths, with analytic gradients from exact length derivatives of the
  covariances (`ProductCovariance::matrix_with_length_derivatives` in Rust).
- `VarianceProfile`: a spline standard deviation along a covariance dimension (non-stationary
  variance), with exact covariances and exact gradients for fitting its coefficients;
  `Element::linear_combination` and three-factor inner products in Rust.
- `Space.harmonics` (cos/sin of a period, e.g. zonal waves) and `Space.bsplines_periodic`
  (B-splines that wrap around a period), exact in every integral, for periodic mean structure.
- `rebasis.xarray.optimal_estimate` and `rebasis.xarray.fit_covariance`: estimation from
  observations along one dimension of a Dataset or DataArray, with coordinates by name, datetimes
  (`numeric_time`) and datetime target edges, other dimensions as columns, and a labelled Dataset
  of estimates with CF bounds.

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
