# rebasis

**Linear maps between function bases**, with a Rust core and a Python interface: instrument
line shapes, regridding, conservative up- and down-sampling, and projection.

A quantity is represented as $f(x) = \sum_j f_j B_j(x)$ for some set of basis functions $B_j$.
rebasis computes the sparse matrices that turn the coefficients $f_j$ into

| operation | computes | typical use |
|---|---|---|
| {func}`~rebasis.measure` | $g_i = \int K_i f \,/ \int K_i$ | an instrument line shape applied to a high-resolution spectrum |
| {func}`~rebasis.measure_derivative` | $\partial g_i / \partial \theta_i$ | Jacobians with respect to shift, width, shape parameters |
| {func}`~rebasis.evaluate` | $g_i = f(x_i)$ | interpolation, plotting a reconstruction |
| {class}`~rebasis.Projection` | the best $L^2$ fit $\sum_i g_i G_i \approx f$ | conservative regridding, optionally bounded |
| {class}`~rebasis.Fit` | model coefficients whose measurements match data | smooth conservative upsampling, smoothing splines |
| {func}`~rebasis.optimal_estimate` | best linear unbiased estimates of averages, with uncertainties | monthly zonal means from sparse satellite sampling |

Inner products are exact for piecewise polynomials (boxes, hats, B-splines, tabulated shapes)
and accurate to rounding for the analytic line shapes (Gaussian, Lorentzian, Voigt, pseudo-Voigt,
super-Gaussian, smoothed box, Fourier-transform spectrometer shapes); there are no tolerance or
resolution parameters to tune.

```python
import numpy as np
import rebasis as rb

nu = np.linspace(19_000, 21_000, 200_001)  # model grid in wavenumber [cm⁻¹]
hires = rb.Space.hats(nu)  # the spectrum is the linear interpolant
ils = rb.Space.gaussians(np.arange(480, 520, 0.1), fwhm=0.5)  # instrument, in nm
W = rb.measure(ils.to_wavenumber(jacobian=True), hires)  # scipy.sparse, exact
measured = W @ radiance
```

```{toctree}
:maxdepth: 2

installation
quickstart
concepts
lineshapes
coordinates
regridding
derivatives
estimation
api
rust
```
