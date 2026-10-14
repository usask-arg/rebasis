"""Linear maps between spaces."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from . import _rebasis
from ._space import Space, _vec


def _weight_scale(weight) -> float | None:
    """None (uniform), ``"cos_lat_deg"`` / ``"cos_lat_rad"`` for area weighting in latitude, or
    a float s for the weight cos(s·x)."""
    if weight is None:
        return None
    if isinstance(weight, str):
        if weight in ("cos_lat_deg", "cos_lat_degrees"):
            return math.pi / 180.0
        if weight in ("cos_lat_rad", "cos_lat_radians"):
            return 1.0
        msg = f"unknown weight {weight!r}; use None, 'cos_lat_deg', 'cos_lat_rad' or a scale"
        raise ValueError(msg)
    return float(weight)


def _csr(parts) -> sp.csr_array:
    indptr, indices, data, shape = parts
    return sp.csr_array((data, indices, indptr), shape=shape)


def _to_rows(f, axis: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Moves ``axis`` last and flattens the rest: (k, n) contiguous, plus the leading shape."""
    moved = np.moveaxis(np.asarray(f, dtype=np.float64), axis, -1)
    lead = moved.shape[:-1]
    return np.ascontiguousarray(moved.reshape(-1, moved.shape[-1])), lead


def _from_rows(out: np.ndarray, lead: tuple[int, ...], axis: int) -> np.ndarray:
    return np.moveaxis(out.reshape(*lead, out.shape[-1]), -1, axis)


def inner_products(a: Space, b: Space, weight=None) -> sp.csr_array:
    """The matrix ⟨a_i, b_j⟩ = ∫ a_i(x) b_j(x) w(x) dx, shape (len(a), len(b))."""
    return _csr(_rebasis.inner_products(a._inner, b._inner, _weight_scale(weight)))


def gram(a: Space, weight=None) -> sp.csr_array:
    """The Gram (mass) matrix ⟨a_i, a_j⟩."""
    return inner_products(a, a, weight)


def measure(
    kernels: Space, basis: Space, normalization: str = "integral", weight=None
) -> sp.csr_array:
    """The matrix W with g = W @ f the kernel responses to f = Σ_j f_j B_j.

    Parameters
    ----------
    normalization
        ``"integral"``: g_i = ∫ K_i f w / ∫ K_i w, kernels are unit-area instrument responses.
        ``"rowsum"``: each row of W sums to one; renormalizes kernels that run off the end of
        the basis, and gives weighted averages for a delta basis.
        ``"none"``: plain inner products ⟨K_i, B_j⟩.
    weight
        See :func:`project`.
    """
    parts = _rebasis.measure(kernels._inner, basis._inner, normalization, _weight_scale(weight))
    return _csr(parts)


def evaluate(basis: Space, x) -> sp.csr_array:
    """The matrix E with E @ f = f(x), shape (len(x), len(basis))."""
    return _csr(_rebasis.evaluate(basis._inner, _vec(x, "x")))


def apply_along(matrix, f, axis: int = -1) -> np.ndarray:
    """Applies a sparse matrix, :class:`Projection` or :class:`Fit` along ``axis`` of ``f``."""
    if isinstance(matrix, Projection | Fit):
        return matrix.apply(f, axis=axis)
    rows, lead = _to_rows(f, axis)
    return _from_rows((matrix @ rows.T).T, lead, axis)


class Projection:
    """L2 (Galerkin) projection of f = Σ f_j S_j onto ``dst``: solves ⟨G, G⟩ g = ⟨G, S⟩ f.

    Conserves ∫ f w whenever ``dst`` can represent a constant (boxes, hats, clamped B-splines
    covering the domain of f). For boxes it is conservative (area-weighted) regridding; onto
    deltas it is interpolation; onto the same space it is the identity. The map is usually
    dense, so it is kept factored; use :meth:`to_dense` to get it as a matrix.

    Parameters
    ----------
    weight
        Weight of the inner product: None, ``"cos_lat_deg"`` or ``"cos_lat_rad"`` (area
        weighting for latitude), or a float s for cos(s·x).
    """

    def __init__(self, src: Space, dst: Space, weight=None):
        self._inner = _rebasis.Projection(src._inner, dst._inner, _weight_scale(weight))

    @property
    def shape(self) -> tuple[int, int]:
        return (self._inner.n_dst, self._inner.n_src)

    def apply(self, f, axis: int = -1, bounded: bool = False) -> np.ndarray:
        """Projects ``f`` along ``axis``.

        With ``bounded``, each output stays within the range of the source coefficients it
        overlaps (no new extrema or negative values), with the clipped mass redistributed so
        the integral is unchanged (clip and assured sum). Meaningful when coefficients are
        values: boxes, hats, B-splines.
        """
        rows, lead = _to_rows(f, axis)
        return _from_rows(self._inner.apply_last_axis(rows, bounded), lead, axis)

    __call__ = apply

    def __matmul__(self, f):
        return self.apply(f, axis=0)

    def to_dense(self) -> np.ndarray:
        """The projection matrix, shape (len(dst), len(src))."""
        return self._inner.to_dense()

    @property
    def cross(self) -> sp.csr_array:
        """M = ⟨dst_i, src_j⟩."""
        return _csr(self._inner.cross())


def project(src: Space, dst: Space, weight=None) -> Projection:
    """Shorthand for :class:`Projection`."""
    return Projection(src, dst, weight)


class Fit:
    """Penalized least-squares fit of coefficients in ``model`` to data measured by ``kernels``.

    Minimizes ‖W c − d‖² + λ P(c) with W = measure(kernels, model). With box kernels and a
    spline model this is smooth conservative upsampling (histopolation: the fitted function's
    bin averages reproduce the data); with delta kernels it is a smoothing spline; with
    instrument line shapes it inverts :func:`measure`.

    Parameters
    ----------
    smoothing
        Dimensionless: λ = smoothing · tr(WᵀW) / tr(P). Needed whenever the data do not
        determine every coefficient (e.g. n bins and n + 2 quadratic B-splines); small values
        (1e-6) then pick the smoothest exact fit.
    penalty
        ``"derivative"``: P = ∫ (f⁽ʳ⁾)² dx, a true smoothing spline, blind to polynomials of
        degree < r (needs a model of degree >= r). ``"difference"``: P = Σ (Δʳ c)², the
        P-spline penalty on coefficient differences; r = 0 is ridge regression.
    order
        r.
    """

    def __init__(
        self,
        kernels: Space,
        model: Space,
        *,
        normalization: str = "integral",
        smoothing: float = 0.0,
        penalty: str = "derivative",
        order: int = 2,
        weight=None,
    ):
        self._inner = _rebasis.Fit(
            kernels._inner,
            model._inner,
            normalization,
            float(smoothing),
            penalty,
            int(order),
            _weight_scale(weight),
        )
        self._model = model

    @property
    def shape(self) -> tuple[int, int]:
        return (self._inner.n_model, self._inner.n_data)

    @property
    def model(self) -> Space:
        return self._model

    @property
    def design(self) -> sp.csr_array:
        """W = measure(kernels, model); the fit's predicted data are ``design @ c``."""
        return _csr(self._inner.design())

    def apply(self, data, axis: int = -1) -> np.ndarray:
        """Model coefficients fitted to ``data`` along ``axis``."""
        rows, lead = _to_rows(data, axis)
        return _from_rows(self._inner.apply_last_axis(rows), lead, axis)

    __call__ = apply

    def to_dense(self) -> np.ndarray:
        """The linear map from data to coefficients, shape (len(model), len(kernels))."""
        return self._inner.to_dense()


def fit(kernels: Space, model: Space, data, axis: int = -1, **kwargs) -> np.ndarray:
    """Shorthand for ``Fit(kernels, model, **kwargs).apply(data, axis)``."""
    return Fit(kernels, model, **kwargs).apply(data, axis)
