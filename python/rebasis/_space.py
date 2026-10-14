"""Spaces: ordered sets of basis functions or measurement kernels."""

from __future__ import annotations

import numpy as np

from . import _rebasis

FWHM_PER_SIGMA = _rebasis.FWHM_PER_SIGMA


def _vec(x, name: str) -> np.ndarray:
    a = np.ascontiguousarray(np.atleast_1d(np.asarray(x, dtype=np.float64)))
    if a.ndim != 1:
        msg = f"{name} must be 1-D, got shape {a.shape}"
        raise ValueError(msg)
    return a


def _param(value, n: int, name: str) -> np.ndarray:
    """A scalar or per-element parameter, broadcast to length n."""
    a = np.asarray(value, dtype=np.float64)
    if a.ndim > 1 or (a.ndim == 1 and a.size not in (1, n)):
        msg = f"{name} must be a scalar or have one value per element ({n}), got shape {a.shape}"
        raise ValueError(msg)
    return np.ascontiguousarray(np.broadcast_to(a.reshape(-1), (n,)))


def _one_of(name_a: str, a, name_b: str, b, convert_b=lambda v: v):
    if (a is None) == (b is None):
        msg = f"give exactly one of {name_a} and {name_b}"
        raise ValueError(msg)
    return a if a is not None else convert_b(np.asarray(b, dtype=np.float64))


class Space:
    """An ordered set of elements: basis functions B_j, or measurement kernels K_i.

    Elements are shapes, not densities: everything peaks at 1 (boxes, hats, B-spline sums and
    all line-shape profiles). Normalization belongs to the operation: :func:`measure` divides
    by each kernel's integral, so the kernels act as unit-area instrument responses.

    Line-shape profiles are truncated; ``cutoff`` sets the half width of the support in the
    profile's natural width unit (stated per constructor). Within the support every integral
    is accurate to rounding.
    """

    def __init__(self, inner: _rebasis.Space, kind: str = "custom"):
        self._inner = inner
        self._kind = kind

    # ------------------------------------------------------------------ bases

    @classmethod
    def deltas(cls, x) -> Space:
        """Point evaluations at ``x`` (as kernels), or point masses (as a basis)."""
        return cls(_rebasis.Space.deltas(_vec(x, "x")), "deltas")

    @classmethod
    def boxes(cls, edges) -> Space:
        """Piecewise-constant basis; box i is 1 on [edges[i], edges[i+1]). Coefficients are
        bin averages."""
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
        """Piecewise-linear basis on ``nodes``: coefficients are the values at the nodes, and the
        function is their linear interpolant. The basis for a function known by samples."""
        return cls(_rebasis.Space.hats(_vec(nodes, "nodes")), "hats")

    @classmethod
    def bsplines(cls, knots, degree: int) -> Space:
        """B-splines of ``degree`` (0-3) on a nondecreasing knot vector:
        len(knots) − degree − 1 functions."""
        return cls(_rebasis.Space.bsplines(_vec(knots, "knots"), int(degree)), f"bsplines{degree}")

    @classmethod
    def bsplines_clamped(cls, breakpoints, degree: int = 3) -> Space:
        """B-splines of ``degree`` (0-3) with repeated end knots: a partition of unity on
        [breakpoints[0], breakpoints[-1]] with len(breakpoints) + degree − 1 functions. Degree
        0 is :meth:`boxes`, degree 1 :meth:`hats`."""
        inner = _rebasis.Space.bsplines_clamped(_vec(breakpoints, "breakpoints"), int(degree))
        return cls(inner, f"bsplines{degree}")

    # ------------------------------------------------------------ line shapes

    @classmethod
    def gaussians(cls, centers, *, sigma=None, fwhm=None, cutoff: float = 6.0) -> Space:
        """exp(−t²/2σ²); ``cutoff`` in standard deviations."""
        c = _vec(centers, "centers")
        s = _one_of("sigma", sigma, "fwhm", fwhm, lambda f: f / FWHM_PER_SIGMA)
        inner = _rebasis.Space.gaussians(c, _param(s, len(c), "sigma"), float(cutoff))
        return cls(inner, "gaussians")

    @classmethod
    def lorentzians(cls, centers, *, hwhm=None, fwhm=None, cutoff: float = 100.0) -> Space:
        """1 / (1 + (t/γ)²), γ = HWHM; ``cutoff`` in HWHM. The heavy tails are truncated, and
        :func:`measure` normalizes by the truncated area."""
        c = _vec(centers, "centers")
        h = _one_of("hwhm", hwhm, "fwhm", fwhm, lambda f: f / 2)
        inner = _rebasis.Space.lorentzians(c, _param(h, len(c), "hwhm"), float(cutoff))
        return cls(inner, "lorentzians")

    @classmethod
    def voigts(
        cls,
        centers,
        *,
        sigma=None,
        fwhm_gaussian=None,
        gamma=None,
        fwhm_lorentzian=None,
        cutoff: float = 100.0,
    ) -> Space:
        """Gaussian (``sigma`` or ``fwhm_gaussian``) convolved with Lorentzian (half width
        ``gamma`` or ``fwhm_lorentzian``), via the Faddeeva function; ``cutoff`` in units of the
        Voigt HWHM."""
        c = _vec(centers, "centers")
        s = _one_of("sigma", sigma, "fwhm_gaussian", fwhm_gaussian, lambda f: f / FWHM_PER_SIGMA)
        g = _one_of("gamma", gamma, "fwhm_lorentzian", fwhm_lorentzian, lambda f: f / 2)
        n = len(c)
        inner = _rebasis.Space.voigts(
            c, _param(s, n, "sigma"), _param(g, n, "gamma"), float(cutoff)
        )
        return cls(inner, "voigts")

    @classmethod
    def pseudo_voigts(cls, centers, *, eta, hwhm=None, fwhm=None, cutoff: float = 100.0) -> Space:
        """η · Lorentzian + (1 − η) · Gaussian with a common HWHM; ``cutoff`` in HWHM."""
        c = _vec(centers, "centers")
        h = _one_of("hwhm", hwhm, "fwhm", fwhm, lambda f: f / 2)
        n = len(c)
        inner = _rebasis.Space.pseudo_voigts(
            c, _param(h, n, "hwhm"), _param(eta, n, "eta"), float(cutoff)
        )
        return cls(inner, "pseudo_voigts")

    @classmethod
    def super_gaussians(
        cls,
        centers,
        *,
        power,
        hwhm=None,
        fwhm=None,
        asymmetry=0.0,
        cutoff: float = 6.0,
    ) -> Space:
        """exp(−ln2 · |t / w|^power), flat-topped for power > 2 (power >= 1).

        With ``asymmetry`` a, the half widths are w(1 − a) left and w(1 + a) right of the
        center (the FWHM stays 2w), as used for e.g. TROPOMI slit functions. ``cutoff`` in
        half widths of the respective side.
        """
        c = _vec(centers, "centers")
        h = _one_of("hwhm", hwhm, "fwhm", fwhm, lambda f: f / 2)
        n = len(c)
        h = _param(h, n, "hwhm")
        a = _param(asymmetry, n, "asymmetry")
        if np.any(np.abs(a) >= 1):
            msg = "asymmetry must be in (−1, 1)"
            raise ValueError(msg)
        inner = _rebasis.Space.super_gaussians(
            c,
            np.ascontiguousarray(h * (1 - a)),
            np.ascontiguousarray(h * (1 + a)),
            _param(power, n, "power"),
            float(cutoff),
        )
        return cls(inner, "super_gaussians")

    @classmethod
    def smoothed_boxes(cls, centers, *, width, sigma=None, fwhm=None, cutoff: float = 6.0) -> Space:
        """A box of full ``width`` convolved with a Gaussian: an entrance slit imaged through
        blurring optics. ``cutoff`` in standard deviations beyond the box edges."""
        c = _vec(centers, "centers")
        s = _one_of("sigma", sigma, "fwhm", fwhm, lambda f: f / FWHM_PER_SIGMA)
        n = len(c)
        inner = _rebasis.Space.smoothed_boxes(
            c,
            np.ascontiguousarray(_param(width, n, "width") / 2),
            _param(s, n, "sigma"),
            float(cutoff),
        )
        return cls(inner, "smoothed_boxes")

    @classmethod
    def fts(cls, centers, *, max_opd, apodization="none", cutoff: float = 100.0) -> Space:
        """Fourier-transform spectrometer line shape, ∫ A(u) cos(2πL(ν − ν₀)u) du.

        ``max_opd`` L is the maximum optical path difference in inverse units of the spectral
        coordinate (cm for wavenumbers in cm⁻¹). ``apodization`` is one of ``"none"`` (sinc),
        ``"triangle"`` (sinc²), ``"hann"``, ``"hamming"`` (Happ–Genzel), ``"blackman"``,
        ``"blackman_harris"``, ``"norton_beer_weak"``, ``"norton_beer_medium"``,
        ``"norton_beer_strong"``, or a dict ``{"cosine_sum": [a0, a1, ...]}`` for
        A(u) = Σ a_m cos(mπu) or ``{"polynomial": [c0, c1, ...]}`` for A(u) = Σ c_i (1 − u²)^i.
        ``cutoff`` in units of 1/(2L), the spacing of the sinc zeros.
        """
        c = _vec(centers, "centers")
        opd = _param(max_opd, len(c), "max_opd")
        if isinstance(apodization, str):
            inner = _rebasis.Space.fts(c, opd, float(cutoff), apodization=apodization)
        elif isinstance(apodization, dict) and len(apodization) == 1:
            ((key, coeffs),) = apodization.items()
            if key not in ("cosine_sum", "polynomial"):
                msg = f"unknown apodization form {key!r}"
                raise ValueError(msg)
            coeffs = [float(v) for v in coeffs]
            inner = _rebasis.Space.fts(c, opd, float(cutoff), **{key: coeffs})
        else:
            msg = "apodization must be a name or a dict {'cosine_sum' | 'polynomial': coeffs}"
            raise TypeError(msg)
        return cls(inner, "fts")

    @classmethod
    def rectangles(cls, centers, *, width) -> Space:
        """Boxes of full ``width`` centred on ``centers``."""
        c = _vec(centers, "centers")
        return cls(_rebasis.Space.rectangles(c, _param(width, len(c), "width")), "rectangles")

    @classmethod
    def triangles(cls, centers, *, hwhm=None, fwhm=None) -> Space:
        """Symmetric triangles with the given half width at half maximum (base 4·hwhm)."""
        c = _vec(centers, "centers")
        h = _one_of("hwhm", hwhm, "fwhm", fwhm, lambda f: f / 2)
        return cls(_rebasis.Space.triangles(c, _param(h, len(c), "hwhm")), "triangles")

    @classmethod
    def trapezoids(cls, centers, *, top_width, bottom_width) -> Space:
        """1 over the central ``top_width``, falling linearly to 0 at ``bottom_width``."""
        c = _vec(centers, "centers")
        n = len(c)
        inner = _rebasis.Space.trapezoids(
            c,
            np.ascontiguousarray(_param(top_width, n, "top_width") / 2),
            np.ascontiguousarray(_param(bottom_width, n, "bottom_width") / 2),
        )
        return cls(inner, "trapezoids")

    @classmethod
    def shifted(cls, centers, shape_x, shape_y, stretch=1.0, interp: str = "linear") -> Space:
        """Copies of one tabulated line shape, stored once.

        Element i is y((x − centers[i]) / stretch[i]) with y the ``"linear"`` or ``"pchip"``
        (monotone cubic) interpolant of (shape_x, shape_y), zero outside the table.
        """
        c = _vec(centers, "centers")
        inner = _rebasis.Space.shifted(
            _vec(shape_x, "shape_x"),
            _vec(shape_y, "shape_y"),
            c,
            _param(stretch, len(c), "stretch"),
            interp,
        )
        return cls(inner, "shifted")

    @classmethod
    def tabulated(cls, centers, x, y, interp: str = "linear") -> Space:
        """A separately tabulated line shape per element (e.g. measured slit functions).

        ``y`` has shape (len(centers), m); ``x`` holds offsets from the centers, shape (m,) if
        shared or (len(centers), m).
        """
        c = _vec(centers, "centers")
        x = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float64)))
        y = np.ascontiguousarray(np.atleast_2d(np.asarray(y, dtype=np.float64)))
        return cls(_rebasis.Space.tabulated(c, x, y, interp), "tabulated")

    # ---------------------------------------------------- coordinate changes

    def to_wavenumber(self, *, jacobian: bool) -> Space:
        """This space, defined in wavelength (nm), as functions of wavenumber (cm⁻¹).

        ``jacobian`` decides how values carry over; there is no default because it is a
        physical choice:

        - ``True`` multiplies by |dλ/dν|, so that integrals over ν equal integrals over λ. Use
          it for **kernels** (an instrument response given in wavelength averages over
          wavelength, wherever the spectrum is computed) and for bases of **spectral
          densities** (radiance per nm).
        - ``False`` keeps values, f(ν) = f(λ(ν)): bases of **functions** (transmittance,
          reflectance, sun-normalized radiance).

        Shapes are exact: a Gaussian in wavelength is integrated as such in wavenumber.
        Parameters (centers, widths) stay in nm, also for :func:`measure_derivative`.
        """
        return self.reciprocal(1e7, jacobian=jacobian)

    def to_wavelength(self, *, jacobian: bool) -> Space:
        """This space, defined in wavenumber (cm⁻¹), as functions of wavelength (nm); see
        :meth:`to_wavenumber` for ``jacobian``."""
        return self.reciprocal(1e7, jacobian=jacobian)

    def reciprocal(self, k: float, *, jacobian: bool) -> Space:
        """The elements as functions of x = k / u, u their own coordinate (supports must not
        contain 0); see :meth:`to_wavenumber` for ``jacobian``."""
        return Space(self._inner.mapped("reciprocal", float(k), 0.0, jacobian), f"1/({self._kind})")

    def affine(self, offset: float = 0.0, scale: float = 1.0, *, jacobian: bool) -> Space:
        """The elements as functions of x = offset + scale·u, u their own coordinate (unit
        changes, Doppler shifts); see :meth:`to_wavenumber` for ``jacobian``."""
        inner = self._inner.mapped("affine", float(offset), float(scale), jacobian)
        return Space(inner, f"affine({self._kind})")

    # ------------------------------------------------------------- utilities

    def derivative(self, order: int = 1) -> Space:
        """The ``order``-th derivatives of piecewise-polynomial elements (jumps excluded)."""
        return Space(self._inner.derivative(int(order)), f"d{order}({self._kind})")

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return f"Space({self._kind}, empty)"
        return f"Space({self._kind}, n={n}, support=[{self.lower.min():g}, {self.upper.max():g}])"

    @property
    def kind(self) -> str:
        return self._kind

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

    def integrals(self, weight=None) -> np.ndarray:
        """∫ φ_i(x) w(x) dx for each element (w(x₀) for a delta at x₀)."""
        from ._ops import _weight_scale

        return self._inner.integrals(_weight_scale(weight))

    def evaluate(self, x):
        """The design matrix E, E[k, j] = B_j(x_k), so that f(x) = E @ coefficients."""
        from ._ops import evaluate

        return evaluate(self, x)
