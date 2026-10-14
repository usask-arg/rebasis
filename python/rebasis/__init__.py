"""Linear maps between function bases.

A quantity f(x) = Σ_j f_j B_j(x) is described by its coefficients f_j in a :class:`Space`.
rebasis computes the (sparse) matrices that take f_j to

- the responses of measurement kernels K_i, e.g. an instrument line shape applied to a
  high-resolution spectrum: :func:`measure`;
- point values f(x_i): :func:`evaluate`;
- the coefficients g_i of the best (L2) representation Σ_i g_i G_i(x) of f in another space,
  for regridding: :class:`Projection`.

Elements are shapes, not densities: boxes, hats and Gaussians peak at 1. A grid of hats or
boxes is a partition of unity, so its coefficients are function values. Use hats for a function
known by samples on a grid (it is then the linear interpolant of the samples), and boxes for
bin averages.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from . import _rebasis
from ._rebasis import SingularError

__version__ = _rebasis.__version__

__all__ = [
    "Projection",
    "SingularError",
    "Space",
    "apply_along",
    "evaluate",
    "gram",
    "inner_products",
    "measure",
    "project",
]

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _vec(x, name: str) -> np.ndarray:
    a = np.ascontiguousarray(np.atleast_1d(np.asarray(x, dtype=np.float64)))
    if a.ndim != 1:
        msg = f"{name} must be 1-D, got shape {a.shape}"
        raise ValueError(msg)
    return a


def _csr(parts) -> sp.csr_array:
    indptr, indices, data, shape = parts
    return sp.csr_array((data, indices, indptr), shape=shape)


class Space:
    """An ordered set of elements: basis functions B_j, or measurement kernels K_i."""

    def __init__(self, inner: _rebasis.Space, kind: str = "custom"):
        self._inner = inner
        self._kind = kind

    @classmethod
    def deltas(cls, x) -> Space:
        """Point evaluations at ``x`` (as kernels), or point masses (as a basis)."""
        return cls(_rebasis.Space.deltas(_vec(x, "x")), "deltas")

    @classmethod
    def boxes(cls, edges) -> Space:
        """Piecewise-constant basis; box i is 1 on [edges[i], edges[i+1])."""
        return cls(_rebasis.Space.boxes(_vec(edges, "edges")), "boxes")

    @classmethod
    def boxes_from_centers(cls, centers) -> Space:
        """Boxes with edges halfway between ``centers``, the end boxes symmetric about theirs."""
        c = _vec(centers, "centers")
        if len(c) < 2:
            msg = "boxes_from_centers needs at least 2 centers"
            raise ValueError(msg)
        mid = 0.5 * (c[1:] + c[:-1])
        edges = np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])
        return cls.boxes(edges)

    @classmethod
    def hats(cls, nodes) -> Space:
        """Piecewise-linear basis on ``nodes``: coefficients are the values at the nodes."""
        return cls(_rebasis.Space.hats(_vec(nodes, "nodes")), "hats")

    @classmethod
    def gaussians(cls, centers, *, sigma=None, fwhm=None, cutoff: float = 6.0) -> Space:
        """Gaussians at ``centers``, truncated at ``cutoff`` standard deviations.

        Give exactly one of ``sigma`` and ``fwhm``, as a scalar or one value per center.
        """
        if (sigma is None) == (fwhm is None):
            msg = "give exactly one of sigma and fwhm"
            raise ValueError(msg)
        s = _vec(sigma, "sigma") if sigma is not None else _vec(fwhm, "fwhm") / FWHM_PER_SIGMA
        inner = _rebasis.Space.gaussians(_vec(centers, "centers"), s, float(cutoff))
        return cls(inner, "gaussians")

    @classmethod
    def shifted(cls, centers, shape_x, shape_y, stretch=1.0) -> Space:
        """Copies of a tabulated line shape.

        Element i is y((x − centers[i]) / stretch[i]) with y the piecewise-linear interpolant of
        (shape_x, shape_y), zero outside [shape_x[0], shape_x[-1]]. The shape is stored once.
        """
        inner = _rebasis.Space.shifted_linear(
            _vec(shape_x, "shape_x"),
            _vec(shape_y, "shape_y"),
            _vec(centers, "centers"),
            _vec(stretch, "stretch"),
        )
        return cls(inner, "shifted")

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return f"Space({self._kind}, empty)"
        return f"Space({self._kind}, n={n}, support=[{self.lower.min():g}, {self.upper.max():g}])"

    @property
    def lower(self) -> np.ndarray:
        """Lower end of each element's support."""
        return self._inner.lower()

    @property
    def upper(self) -> np.ndarray:
        """Upper end of each element's support."""
        return self._inner.upper()

    @property
    def centers(self) -> np.ndarray:
        return self._inner.centers()

    @property
    def integrals(self) -> np.ndarray:
        """∫ φ_i(x) dx for each element (1 for a delta)."""
        return self._inner.integrals()

    def evaluate(self, x) -> sp.csr_array:
        """The design matrix E, E[k, j] = B_j(x_k), so that f(x) = E @ coefficients."""
        return evaluate(self, x)


def inner_products(a: Space, b: Space) -> sp.csr_array:
    """The matrix ⟨a_i, b_j⟩ = ∫ a_i(x) b_j(x) dx, shape (len(a), len(b))."""
    return _csr(_rebasis.inner_products(a._inner, b._inner))


def gram(a: Space) -> sp.csr_array:
    """The Gram (mass) matrix ⟨a_i, a_j⟩."""
    return inner_products(a, a)


def measure(kernels: Space, basis: Space, normalization: str = "integral") -> sp.csr_array:
    """The matrix W with g = W @ f the kernel responses to f = Σ_j f_j B_j.

    Parameters
    ----------
    normalization
        ``"integral"``: g_i = ∫ K_i f / ∫ K_i, kernels are unit-area instrument responses.
        ``"rowsum"``: each row of W sums to one; renormalizes kernels that run off the end of
        the basis, and gives weighted averages for a delta basis.
        ``"none"``: plain inner products ⟨K_i, B_j⟩.
    """
    return _csr(_rebasis.measure(kernels._inner, basis._inner, normalization))


def evaluate(basis: Space, x) -> sp.csr_array:
    """The matrix E with E @ f = f(x), shape (len(x), len(basis))."""
    return _csr(_rebasis.evaluate(basis._inner, _vec(x, "x")))


def apply_along(matrix, f, axis: int = -1) -> np.ndarray:
    """Applies a sparse matrix (or :class:`Projection`) along ``axis`` of an n-d array."""
    if isinstance(matrix, Projection):
        return matrix.apply(f, axis=axis)
    f = np.asarray(f)
    moved = np.moveaxis(f, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = (matrix @ flat.T).T
    return np.moveaxis(out.reshape(*moved.shape[:-1], matrix.shape[0]), -1, axis)


class Projection:
    """L2 (Galerkin) projection of f = Σ f_j S_j onto ``dst``: solves ⟨G, G⟩ g = ⟨G, S⟩ f.

    Conserves ∫ f whenever ``dst`` can represent a constant (boxes, hats covering the domain of
    f). For boxes it is conservative area-weighted regridding; onto deltas it is interpolation;
    onto the same space it is the identity. The map G⁻¹M is usually dense, so it is kept
    factored; use :meth:`to_dense` to get it as a matrix.
    """

    def __init__(self, src: Space, dst: Space):
        self._inner = _rebasis.Projection(src._inner, dst._inner)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._inner.n_dst, self._inner.n_src)

    def apply(self, f, axis: int = -1) -> np.ndarray:
        """Projects ``f`` along ``axis`` (the coefficient axis)."""
        f = np.asarray(f, dtype=np.float64)
        moved = np.moveaxis(f, axis, -1)
        flat = np.ascontiguousarray(moved.reshape(-1, moved.shape[-1]))
        out = self._inner.apply_last_axis(flat)
        return np.moveaxis(out.reshape(*moved.shape[:-1], self.shape[0]), -1, axis)

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


def project(src: Space, dst: Space) -> Projection:
    """Shorthand for :class:`Projection` (src, dst)."""
    return Projection(src, dst)
