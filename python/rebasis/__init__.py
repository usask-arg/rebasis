"""Linear maps between function bases.

A quantity f(x) = Σ_j f_j B_j(x) is described by its coefficients f_j in a :class:`Space`.
rebasis computes the (sparse) matrices that take f_j to

- the responses of measurement kernels K_i, e.g. an instrument line shape applied to a
  high-resolution spectrum: :func:`measure`;
- point values f(x_i): :func:`evaluate`;
- the coefficients g_i of the best (L2) representation Σ_i g_i G_i(x) of f in another space,
  for regridding: :class:`Projection`;
- the coefficients of a model whose measurements match given data, for smooth conservative
  upsampling and smoothing: :class:`Fit`;
- optimal estimates of grid averages (e.g. monthly zonal means) from irregularly sampled
  observations, with uncertainties and averaging kernels: :func:`optimal_estimate`.

Elements are shapes, not densities: boxes, hats, B-splines and every line-shape profile peak
at 1. A grid of hats, boxes or clamped B-splines is a partition of unity, so its coefficients
are function values. Use hats for a function known by samples on a grid (it is then the linear
interpolant of the samples), and boxes for bin averages.

N-dimensional regridding of arrays is :func:`regrid` / :func:`regrid_nd`; for xarray objects
see :func:`rebasis.xarray.regrid`.
"""

from __future__ import annotations

from . import _rebasis
from ._estimate import (
    Covariance,
    CovarianceDim,
    CovarianceSum,
    Estimate,
    MeanTerm,
    VarianceProfile,
    log_likelihood,
    optimal_estimate,
)
from ._fit import CovarianceFit, fit_covariance
from ._ops import (
    PARAMETERS,
    Fit,
    Projection,
    apply_along,
    evaluate,
    fit,
    gram,
    inner_products,
    measure,
    measure_derivative,
    project,
)
from ._rebasis import SingularError, faddeeva_re
from ._regrid import regrid, regrid_nd
from ._space import FWHM_PER_SIGMA, Space

__version__ = _rebasis.__version__


def __getattr__(name: str):
    # rebasis.xarray imports xarray, so load it only when used
    if name == "xarray":
        import importlib

        return importlib.import_module(".xarray", __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "FWHM_PER_SIGMA",
    "PARAMETERS",
    "Covariance",
    "CovarianceDim",
    "CovarianceFit",
    "CovarianceSum",
    "Estimate",
    "Fit",
    "MeanTerm",
    "Projection",
    "SingularError",
    "Space",
    "VarianceProfile",
    "apply_along",
    "evaluate",
    "faddeeva_re",
    "fit",
    "fit_covariance",
    "gram",
    "inner_products",
    "log_likelihood",
    "measure",
    "measure_derivative",
    "optimal_estimate",
    "project",
    "regrid",
    "regrid_nd",
]
