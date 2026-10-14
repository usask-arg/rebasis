---
file_format: mystnb
kernelspec:
  name: python3
---

# Wavelength and wavenumber

Radiative transfer models often compute spectra on a wavenumber grid while instrument line
shapes are characterized in wavelength. A Gaussian in wavelength is not a Gaussian in
wavenumber, so rather than approximating shapes, {meth}`~rebasis.Space.to_wavenumber` (and
{meth}`~rebasis.Space.to_wavelength`, {meth}`~rebasis.Space.reciprocal`,
{meth}`~rebasis.Space.affine`) use each element in the other coordinate exactly: knots are
mapped and the integrator follows the shape through the change of variable.

## The Jacobian

`jacobian` is required, because it is a physical choice:

- `jacobian=True` multiplies by $|d\lambda/d\nu|$ so that integrals over $\nu$ equal integrals
  over $\lambda$. Use it for **kernels**: an instrument response given in wavelength averages
  over wavelength, whatever grid the spectrum is computed on. Use it also for bases of
  **spectral densities** (radiance per nm).
- `jacobian=False` keeps values, $f(\nu) = f(\lambda(\nu))$: bases of **functions**
  (transmittance, reflectance, sun-normalized radiance).

## Example

A model spectrum on a wavenumber grid, measured by a spectrometer whose line shape is a 0.5 nm
Gaussian in wavelength:

```{code-cell} ipython3
import numpy as np

import rebasis as rb

nu = np.linspace(19_500, 20_500, 100_001)          # cm⁻¹
f = 1 + 0.3 * np.sin(nu / 5.0)                     # e.g. a transmittance
hires = rb.Space.hats(nu)

centers_nm = np.arange(490.0, 510.0, 0.2)
ils = rb.Space.gaussians(centers_nm, fwhm=0.5).to_wavenumber(jacobian=True)
measured = rb.measure(ils, hires) @ f
measured[:5]
```

The same measurement computed the other way round, in wavelength with the basis mapped instead,
agrees to the conditioning of the coordinates:

```{code-cell} ipython3
other = rb.measure(rb.Space.gaussians(centers_nm, fwhm=0.5), hires.to_wavelength(jacobian=False)) @ f
print(f"max difference: {np.abs(measured - other).max():.1e}")
```

Parameters stay in the kernel's own coordinate: {func}`~rebasis.measure_derivative` with
`"center"` gives derivatives with respect to the line center in nm, and `"scale"` with respect
to the width in nm.
