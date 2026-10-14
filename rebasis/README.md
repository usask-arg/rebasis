# rebasis

Linear maps between function bases: instrument line shapes, regridding, projection.

A quantity f(x) = Σ_j f_j B_j(x) is described by coefficients in a `Space` of elements (boxes,
hats, B-splines, delta functions, and line-shape profiles: Gaussian, Lorentzian, Voigt,
pseudo-Voigt, super-Gaussian, smoothed box, Fourier-transform spectrometer line shapes with
the standard apodizations, and tabulated shapes). rebasis computes the sparse matrices that
take f_j to

- the responses of measurement kernels (`measure`), with exact derivatives with respect to the
  kernels' parameters (`measure_derivative`);
- point values (`evaluate`);
- the best L2 representation in another space (`Projection`, optionally bounded), for
  conservative regridding;
- the coefficients of a model fitted to measurements (`Fit`), for smooth conservative
  upsampling and smoothing splines.

Elements can be used in another coordinate (`Space::mapped`, e.g. a line shape given in
wavelength against a basis in wavenumber). Inner products are exact for piecewise polynomials
and accurate to rounding for the analytic profiles.

The Python package is [`pyrebasis`](https://pypi.org/project/pyrebasis/) (`import rebasis`);
documentation at <https://usask-arg.github.io/rebasis/>.

License: MIT.
