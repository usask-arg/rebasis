"""Optimal estimation of averages from irregularly sampled data.

Observations y_k are averages of an unknown field f over product kernels (points, footprints,
time windows) in a D-dimensional coordinate, with noise. Targets are averages of f over a grid
of product elements (e.g. latitude band × all longitudes × month), optionally area weighted.
With a Gaussian-process prior of separable covariance σ² Π_d c_d and a constant mean, the
best linear unbiased estimate of the targets (kriging) is

    ĝ = β + B Q⁻¹ (y − β),    Q = C_oo + diag(noise),

with B the target–observation covariance and β the generalized-least-squares mean (or a known
mean). rebasis computes every covariance exactly, including those between averages over boxes
(a monthly mean against a point, a month against a month); the dense algebra uses scipy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from . import _rebasis
from ._ops import _weight_scale, measure
from ._space import Space

FAMILIES = ("squared_exponential", "exponential", "matern32", "matern52")


@dataclass(frozen=True, eq=False)
class VarianceProfile:
    """A variance that varies along a dimension: s(x) = Σ_i c_i B_i(x), so that the covariance
    along it is s(x) s(x') ρ(x − x'). With B-spline bases and positive coefficients s stays
    positive. The basis must cover every coordinate it is used at (s is 0 outside it)."""

    basis: Space
    coefficients: np.ndarray

    def __post_init__(self):
        c = np.ascontiguousarray(np.asarray(self.coefficients, dtype=np.float64).ravel())
        if len(c) != len(self.basis):
            msg = f"{len(c)} coefficients for {len(self.basis)} basis functions"
            raise ValueError(msg)
        object.__setattr__(self, "coefficients", c)

    def __call__(self, x) -> np.ndarray:
        """s at x."""
        return self.basis.evaluate(x) @ self.coefficients


@dataclass(frozen=True)
class CovarianceDim:
    """The correlation function along one dimension.

    Parameters
    ----------
    family
        ``"squared_exponential"`` (very smooth fields), or the rougher Matérn family:
        ``"exponential"`` (Matérn ½), ``"matern32"``, ``"matern52"``.
    length
        The correlation length ℓ, in the units of the coordinate.
    period
        For periodic coordinates such as longitude (e.g. 360): correlations wrap around.
    variance_profile
        A :class:`VarianceProfile` s: the covariance along this dimension becomes
        s(x) s(x') ρ(x − x'), e.g. a variability that peaks at the polar vortex edge.
    """

    family: str
    length: float
    period: float | None = None
    variance_profile: VarianceProfile | None = None

    def __post_init__(self):
        if self.family not in FAMILIES and self.family not in ("gaussian", "matern12"):
            msg = f"family must be one of {FAMILIES}, got {self.family!r}"
            raise ValueError(msg)

    def _spec(self):
        period = None if self.period is None else float(self.period)
        vp = self.variance_profile
        profile = None if vp is None else (vp.basis._inner, vp.coefficients.tolist())
        return (self.family, float(self.length), period, profile)


class Covariance:
    """A separable prior covariance σ² Π_d c_d(x_d − x'_d), one :class:`CovarianceDim` per named
    dimension, e.g. ``Covariance(4.0, lat=CovarianceDim("squared_exponential", 10.0), ...)``.
    The keyword order fixes the order of the dimensions.
    """

    def __init__(self, variance: float, **dims: CovarianceDim):
        if not dims:
            msg = "a covariance needs at least one dimension"
            raise ValueError(msg)
        self.variance = float(variance)
        self.dims = dict(dims)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.dims)

    def replace(self, variance: float | None = None, **dims: CovarianceDim) -> Covariance:
        """A copy with some parameters changed (for fitting them)."""
        new = dict(self.dims)
        for k, v in dims.items():
            if k not in new:
                msg = f"unknown dimension {k!r}"
                raise ValueError(msg)
            new[k] = v
        return Covariance(self.variance if variance is None else variance, **new)

    def _spec(self):
        return [d._spec() for d in self.dims.values()]

    def matrix(self, a: _Atoms, b: _Atoms, symmetric: bool = False) -> np.ndarray:
        return _rebasis.covariance_matrix(
            self.variance, self._spec(), a.spaces, a.weights, b.spaces, b.weights, symmetric
        )

    def diagonal(self, a: _Atoms) -> np.ndarray:
        return _rebasis.covariance_diagonal(self.variance, self._spec(), a.spaces, a.weights)

    def __add__(self, other: Covariance | CovarianceSum) -> CovarianceSum:
        return CovarianceSum(self) + other


class CovarianceSum:
    """A sum of separable covariances over the same dimensions, e.g. a small-scale and a
    large-scale component: ``Covariance(...) + Covariance(...)``. Each component is computed
    exactly; the sum is not separable."""

    def __init__(self, *components: Covariance):
        if not components:
            msg = "a covariance sum needs at least one component"
            raise ValueError(msg)
        names = components[0].names
        if any(c.names != names for c in components):
            msg = "every component must have the same dimensions, in the same order"
            raise ValueError(msg)
        self.components = list(components)

    @property
    def names(self) -> tuple[str, ...]:
        return self.components[0].names

    def __add__(self, other: Covariance | CovarianceSum) -> CovarianceSum:
        more = other.components if isinstance(other, CovarianceSum) else [other]
        return CovarianceSum(*self.components, *more)

    def matrix(self, a: _Atoms, b: _Atoms, symmetric: bool = False) -> np.ndarray:
        return sum(c.matrix(a, b, symmetric) for c in self.components)

    def diagonal(self, a: _Atoms) -> np.ndarray:
        return sum(c.diagonal(a) for c in self.components)


@dataclass
class _Atoms:
    spaces: list
    weights: list
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return math.prod(self.shape)


def _observation_atoms(coords: Mapping, names, n: int | None) -> _Atoms:
    missing = set(names) - set(coords)
    extra = set(coords) - set(names)
    if missing or extra:
        msg = f"coords must have exactly the covariance's dimensions {names}"
        raise ValueError(msg)
    spaces = []
    for name in names:
        c = coords[name]
        s = c if isinstance(c, Space) else Space.deltas(np.asarray(c, dtype=np.float64).ravel())
        if n is not None and len(s) != n:
            msg = f"coords[{name!r}] has {len(s)} entries, expected {n}"
            raise ValueError(msg)
        n = len(s)
        spaces.append(s._inner)
    return _Atoms(spaces, [None] * len(names), (n,))


def _grid_atoms(targets: Mapping, names, weight: Mapping | None) -> _Atoms:
    if set(targets) != set(names):
        msg = f"targets must have exactly the covariance's dimensions {names}"
        raise ValueError(msg)
    weight = dict(weight or {})
    shape = tuple(len(targets[name]) for name in names)
    idx = np.unravel_index(np.arange(math.prod(shape)), shape)
    spaces = [targets[name]._inner.take(idx[d].tolist()) for d, name in enumerate(names)]
    return _Atoms(spaces, [_weight_scale(weight.get(name)) for name in names], shape)


@dataclass(frozen=True)
class MeanTerm:
    """One term of a structured mean: a tensor product of spaces, constant along dimensions not
    given (``{}`` is a constant).

    Parameters
    ----------
    bases
        Dimension name -> :class:`Space`.
    prior_std
        The prior standard deviation of the term's coefficients (a scalar, or one per
        coefficient). ``inf`` (the default) leaves them unconstrained: estimated by generalized
        least squares, and reproduced exactly however the data are sampled. A finite value is a
        zero-mean Gaussian prior that shrinks coefficients the data do not determine, e.g. a
        drift through the month where only part of it was sampled, or latitude structure
        beyond the data. Use ``inf`` rather than a huge finite value for "unconstrained": a
        finite prior enters the covariance, and one far larger than the field's variance costs
        precision in its factorization.
    """

    bases: Mapping[str, Space]
    prior_std: float | np.ndarray = math.inf


def _parse_mean(mean) -> tuple[list[MeanTerm], float | None]:
    """The mean's terms, or a known constant."""
    if isinstance(mean, str):
        if mean != "estimate":
            msg = f"mean must be 'estimate', a number, a dict of spaces or MeanTerms; got {mean!r}"
            raise ValueError(msg)
        return [MeanTerm({})], None
    if isinstance(mean, MeanTerm):
        return [mean], None
    if isinstance(mean, Mapping):
        return [MeanTerm(dict(mean))], None
    if isinstance(mean, list | tuple):
        terms = []
        for t in mean:
            if isinstance(t, MeanTerm):
                terms.append(t)
            elif isinstance(t, Mapping):
                terms.append(MeanTerm(dict(t)))
            elif t == "estimate":
                terms.append(MeanTerm({}))
            else:
                msg = f"mean terms must be MeanTerm or dicts of spaces, got {t!r}"
                raise ValueError(msg)
        return terms, None
    return [], float(mean)


def _kernels(coords: Mapping, names) -> dict[str, Space]:
    return {
        name: c if isinstance(c, Space) else Space.deltas(np.asarray(c, dtype=np.float64).ravel())
        for name, c in ((name, coords[name]) for name in names)
    }


def _mean_design(terms, names, kernels, n, targets, target_weight):
    """H for the observations (n × P), for the target grid (m × P, or None) and the prior
    standard deviation of each column. A term's values are a row-wise Kronecker product of
    per-dimension measurements (observations) or a Kronecker product (the target grid)."""
    target_weight = dict(target_weight or {})
    h_obs, h_tgt, std = [], [], []
    for term in terms:
        unknown = set(term.bases) - set(names)
        if unknown:
            msg = f"mean basis for unknown dimensions {sorted(unknown)}"
            raise ValueError(msg)
        ho = np.ones((n, 1))
        ht = np.ones((1, 1))
        for name in names:
            basis = term.bases.get(name)
            if basis is None:
                if targets is not None:
                    ht = np.kron(ht, np.ones((len(targets[name]), 1)))
                continue
            d = measure(kernels[name], basis).toarray()  # (n, P_d)
            ho = (ho[:, :, None] * d[:, None, :]).reshape(n, -1)
            if targets is not None:
                ht = np.kron(
                    ht, measure(targets[name], basis, weight=target_weight.get(name)).toarray()
                )
        p = ho.shape[1]
        sd = np.broadcast_to(np.asarray(term.prior_std, dtype=np.float64), (p,))
        if np.any(sd < 0) or np.any(np.isnan(sd)):
            msg = "prior standard deviations must be >= 0 (inf: unconstrained)"
            raise ValueError(msg)
        h_obs.append(ho)
        h_tgt.append(ht if targets is not None else None)
        std.append(sd)
    if not h_obs:
        m = None if targets is None else math.prod(len(targets[name]) for name in names)
        return np.zeros((n, 0)), None if m is None else np.zeros((m, 0)), np.zeros(0)
    h_t = None if targets is None else np.hstack(h_tgt)
    return np.hstack(h_obs), h_t, np.concatenate(std)


@dataclass
class Estimate:
    """The result of :func:`optimal_estimate`. Arrays have the shape of the target grid, with a
    trailing axis of length k when ``values`` had k columns."""

    values: np.ndarray
    """The estimates ĝ."""
    std: np.ndarray
    """Their posterior standard deviations (including the uncertainty of the mean)."""
    prior_std: np.ndarray
    """The prior standard deviations of the targets (from the covariance and the mean terms
    with finite priors), for comparison with ``std``."""
    mean: float | np.ndarray
    """The estimated constant mean β when the mean is a single unconstrained constant (one per
    column), the known mean if one was given, otherwise NaN; see ``mean_coefficients``."""
    log_likelihood: float | np.ndarray
    """log p(y | covariance), one per column; restricted (REML, in Harville's basis-independent
    form) over the unconstrained mean terms. Maximize it over the covariance parameters to fit
    them."""
    dims: tuple[str, ...]
    mean_coefficients: np.ndarray | None = None
    """The estimated coefficients of the unconstrained mean terms, shape (P,) or (P, k)."""
    covariance: np.ndarray | None = None
    """The posterior covariance of the flattened targets, if requested: (m, m), or (k, m, m)
    when columns have different valid observations."""
    weights: np.ndarray | None = None
    """W, with ĝ = W y + offset: (m, n), or (k, m, n) when columns have different valid
    observations (zero weight on invalid ones). Row m is target m's averaging kernel over the
    observations; it sums to 1 when the mean has an unconstrained constant."""
    offset: np.ndarray | None = None
    n_used: np.ndarray | None = None
    """The number of (valid) observations each estimate used: all valid ones, or those in its
    neighbourhood."""

    def apply(self, values) -> np.ndarray:
        """The same estimator applied to other values at the same observations (shape (n,) or
        (n, k)), e.g. a model field sampled like the data: comparing that with the model's true
        averages measures the sampling error."""
        if self.weights is None:
            msg = "the estimate was computed with weights=False"
            raise ValueError(msg)
        v = np.asarray(values, dtype=np.float64)
        v = np.where(np.isfinite(v), v, 0.0)  # invalid observations carry zero weight
        if self.weights.ndim == 2:
            out = self.weights @ v
            out = out + (self.offset if v.ndim == 1 else self.offset[:, None])
        else:
            if v.ndim != 2 or v.shape[1] != self.weights.shape[0]:
                msg = f"values must have shape (n, {self.weights.shape[0]})"
                raise ValueError(msg)
            out = np.einsum("kmn,nk->mk", self.weights, v) + self.offset.T
        grid = self.prior_std.shape[: len(self.dims)]
        return out.reshape(*grid, *out.shape[1:])

    def weight_histogram(self, coord, edges) -> np.ndarray:
        """Σ of each target's weights over the observations whose coordinate ``coord`` (one value
        per observation) falls in each bin of ``edges``: how the estimate draws on, e.g., each
        day of the month. Shape: target grid (+ k) + (len(edges) − 1,)."""
        if self.weights is None:
            msg = "the estimate was computed with weights=False"
            raise ValueError(msg)
        b = np.digitize(np.asarray(coord, dtype=np.float64), edges) - 1
        nb = len(edges) - 1
        onehot = np.zeros((len(b), nb))
        ok = (b >= 0) & (b < nb)
        onehot[np.flatnonzero(ok), b[ok]] = 1.0
        grid = self.prior_std.shape[: len(self.dims)]
        if self.weights.ndim == 2:
            return (self.weights @ onehot).reshape(*grid, nb)
        out = np.einsum("kmn,nb->mkb", self.weights, onehot)
        return out.reshape(*grid, self.weights.shape[0], nb)


def _logdet_spd(a: np.ndarray) -> float:
    """log|A| of a symmetric positive-definite matrix, by Cholesky (numpy's slogdet can emit
    spurious floating-point warnings with some LAPACK builds)."""
    return 2.0 * float(np.sum(np.log(np.diag(linalg.cholesky(a, lower=True)))))


def _factor(q: np.ndarray):
    try:
        return linalg.cho_factor(q, lower=True, check_finite=False)
    except linalg.LinAlgError as e:
        msg = (
            "the observation covariance is not positive definite; observations at identical "
            "coordinates need noise > 0"
        )
        raise _rebasis.SingularError(msg) from e


def _solve(y, q, hf, known, b=None, hf_t=None, prior_var=None, prior_cov=None, weights=False):
    """The estimate for one set of valid observations and k columns of values y (n, k).

    q is the full observation covariance (including noise and the finite-prior mean terms),
    hf the unconstrained mean columns (n, P_f); b, hf_t, prior_var/prior_cov describe the
    targets (omitted for the likelihood alone)."""
    n = y.shape[0]
    cf = _factor(q)
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    pf = hf.shape[1]
    out = {}
    if pf:
        qi_h = linalg.cho_solve(cf, hf, check_finite=False)
        s = hf.T @ qi_h
        try:
            s_cf = linalg.cho_factor(s, lower=True, check_finite=False)
        except linalg.LinAlgError as e:
            msg = (
                "the observations do not determine every unconstrained mean coefficient: give "
                "those terms a finite prior_std, or use bases covered by the data"
            )
            raise _rebasis.SingularError(msg) from e
        beta = linalg.cho_solve(s_cf, qi_h.T @ y, check_finite=False)  # (P_f, k)
        r = y - hf @ beta
        out["beta"] = beta
    else:
        r = y - (known or 0.0)
    alpha = linalg.cho_solve(cf, r, check_finite=False)
    ll = -0.5 * np.einsum("nk,nk->k", r, alpha) - 0.5 * logdet
    if pf:
        logdet_s = 2.0 * np.sum(np.log(np.diag(s_cf[0])))
        # ½ log|HᵀH| makes the restricted likelihood independent of how the unconstrained mean
        # is parametrized (Harville 1974)
        logdet_hh = _logdet_spd(hf.T @ hf)
        ll += -0.5 * logdet_s + 0.5 * logdet_hh - 0.5 * (n - pf) * math.log(2 * math.pi)
    else:
        ll += -0.5 * n * math.log(2 * math.pi)
    out["ll"] = ll
    if b is None:
        return out

    mu_t = hf_t @ out["beta"] if pf else (known or 0.0)
    out["g"] = mu_t + b @ alpha
    x = linalg.cho_solve(cf, b.T, check_finite=False)  # Q⁻¹ Bᵀ, (n, m)
    var = prior_var - np.einsum("mn,nm->m", b, x)
    if pf:
        rmat = hf_t - b @ qi_h  # (m, P_f)
        s_inv_rt = linalg.cho_solve(s_cf, rmat.T, check_finite=False)
        var += np.einsum("mp,pm->m", rmat, s_inv_rt)
    out["var"] = var
    if prior_cov is not None:
        post = prior_cov - b @ x
        if pf:
            post += rmat @ s_inv_rt
        out["cov"] = post
    if weights:
        w = x.T.copy()
        if pf:
            # W = B Q⁻¹ + R S⁻¹ Hᵀ Q⁻¹: unbiased for every field in the unconstrained span
            w += s_inv_rt.T @ qi_h.T
            out["offset"] = np.zeros(len(out["g"]))
        else:
            out["offset"] = (known or 0.0) * (1.0 - w.sum(axis=1))
        out["w"] = w
    return out


def _columns(values, noise):
    """Values as (n, k), noise as (n, k), whether there was one column, and the groups of
    columns sharing the same valid observations and noise."""
    y = np.asarray(values, dtype=np.float64)
    single = y.ndim == 1
    y = y.reshape(len(y), -1)
    n, k = y.shape
    nz = np.asarray(noise, dtype=np.float64)
    if nz.ndim == 2:
        if nz.shape != (n, k):
            msg = f"noise must be a scalar, (n,) or (n, k) = {(n, k)}, got {nz.shape}"
            raise ValueError(msg)
    else:
        nz = np.broadcast_to(nz.reshape(-1, 1) if nz.ndim == 1 else nz, (n, k))
    if np.any(nz < 0):
        msg = "noise variances must be >= 0"
        raise ValueError(msg)
    valid = np.isfinite(y) & np.isfinite(nz)
    groups: dict[bytes, list[int]] = {}
    for j in range(k):
        key = valid[:, j].tobytes() + np.where(valid[:, j], nz[:, j], 0.0).tobytes()
        groups.setdefault(key, []).append(j)
    return y, nz, valid, single, list(groups.values())


def log_likelihood(values, coords: Mapping, covariance: Covariance, *, noise, mean="estimate"):
    """log p(y | covariance) without computing any targets; see :attr:`Estimate.log_likelihood`.
    Cheap enough to maximize over the covariance parameters with ``scipy.optimize``. ``values``
    may have k columns (one likelihood each)."""
    y, nz, valid, single, groups = _columns(values, noise)
    n = y.shape[0]
    names = covariance.names
    obs = _observation_atoms(coords, names, n)
    terms, known = _parse_mean(mean)
    h, _, sd = _mean_design(terms, names, _kernels(coords, names), n, None, None)
    q0 = covariance.matrix(obs, obs, symmetric=True)
    flat = np.isinf(sd)
    hp = h[:, ~flat] * sd[~flat]
    ll = np.empty(y.shape[1])
    for cols in groups:
        idx = valid[:, cols[0]]
        q = q0[np.ix_(idx, idx)] + np.diag(nz[idx, cols[0]]) + hp[idx] @ hp[idx].T
        ll[cols] = _solve(y[np.ix_(idx, cols)], q, h[idx][:, flat], known)["ll"]
    return float(ll[0]) if single else ll


def _take(atoms: _Atoms, idx: np.ndarray) -> _Atoms:
    sel = idx.tolist()
    return _Atoms([s.take(sel) for s in atoms.spaces], atoms.weights, (len(idx),))


def _period(covariance, name: str) -> float | None:
    comps = covariance.components if isinstance(covariance, CovarianceSum) else [covariance]
    periods = {c.dims[name].period for c in comps} - {None}
    if len(periods) > 1:
        msg = f"the components disagree on the period of {name!r}"
        raise ValueError(msg)
    return periods.pop() if periods else None


def _overlaps(lo, hi, w_lo, w_hi, period):
    """Whether the intervals [lo, hi] (arrays) reach the window [w_lo, w_hi], with wrap-around."""
    if period is None:
        return (lo <= w_hi) & (hi >= w_lo)
    if w_hi - w_lo >= period:
        return np.ones(lo.shape, bool)
    j0 = np.round((0.5 * (w_lo + w_hi) - 0.5 * (lo + hi)) / period)
    out = np.zeros(lo.shape, bool)
    for dj in (-1, 0, 1):
        s = (j0 + dj) * period
        out |= (lo + s <= w_hi) & (hi + s >= w_lo)
    return out


def _blocks(neighborhood, names, targets, tgt: _Atoms, kernels, covariance):
    """(target indices, observation indices) per block of targets sharing a neighbourhood."""
    unknown = set(neighborhood) - set(names)
    if unknown:
        msg = f"neighborhood for unknown dimensions {sorted(unknown)}"
        raise ValueError(msg)
    if any(not (w >= 0) for w in neighborhood.values()):
        msg = "neighborhood half widths must be >= 0"
        raise ValueError(msg)
    dims = [d for d, name in enumerate(names) if name in neighborhood]
    grid_idx = np.unravel_index(np.arange(tgt.size), tgt.shape)
    keys = np.stack([grid_idx[d] for d in dims], axis=1)
    out = []
    for key in np.unique(keys, axis=0):
        t_idx = np.flatnonzero(np.all(keys == key, axis=1))
        mask = np.ones(len(kernels[names[0]]), bool)
        for d, i in zip(dims, key, strict=True):
            name = names[d]
            half = float(neighborhood[name])
            w_lo = targets[name].lower[i] - half
            w_hi = targets[name].upper[i] + half
            k = kernels[name]
            mask &= _overlaps(k.lower, k.upper, w_lo, w_hi, _period(covariance, name))
        out.append((t_idx, np.flatnonzero(mask)))
    return out


def optimal_estimate(
    values,
    coords: Mapping,
    targets: Mapping[str, Space],
    covariance: Covariance,
    *,
    noise,
    mean="estimate",
    target_weight: Mapping | None = None,
    weights: bool = True,
    full_covariance: bool = False,
    neighborhood: Mapping[str, float] | None = None,
) -> Estimate:
    """Optimal (kriging) estimates of grid averages from irregular observations.

    Parameters
    ----------
    values
        The n observations, or an (n, k) array of k sets of values at the same observations
        (e.g. one column per altitude). Columns sharing the same valid (finite) observations
        and noise share one factorization; NaN marks an invalid observation in a column.
    coords
        For each dimension of ``covariance``: the n observation coordinates (point
        observations), or a :class:`Space` of n elements (each observation an average over its
        element, e.g. a footprint or a time window).
    targets
        For each dimension, a :class:`Space` of target elements; the targets are their tensor
        product (e.g. ``{"lat": Space.boxes(lat_edges), "lon": Space.boxes([0, 360]),
        "time": Space.boxes(month_edges)}`` for monthly zonal means).
    covariance
        The prior :class:`Covariance`, or a sum of them (:class:`CovarianceSum`).
    noise
        Observation error variances: a scalar, one per observation (n,), or (n, k).
    mean
        ``"estimate"`` (default): an unknown constant, estimated by generalized least squares
        (ordinary kriging); every row of the weights then sums to 1, so a constant field is
        reproduced exactly however it is sampled. A dict ``{dim: Space}``: a tensor-product mean
        with unconstrained coefficients (universal kriging). A list of :class:`MeanTerm`
        (and dicts): a sum of terms, each unconstrained or with a Gaussian prior that shrinks
        coefficients the data do not determine. Or a known value.
    target_weight
        Per dimension, a weight for the target averages, e.g. ``{"lat": "cos_lat_deg"}``.
    weights, full_covariance
        Also return the weights and the posterior covariance of the targets.
    neighborhood
        Estimate locally: dimension name -> half width. Targets are grouped into blocks by
        their elements along these dimensions (e.g. ``{"lat": 20, "time": 15}`` makes each
        latitude band × month a block), and each block uses only the observations whose
        support lies within the half width of its targets (periodic dimensions wrap). The
        cost then grows with the size of a neighbourhood, not of the whole data set. Mean
        bases that vanish on a block's observations and targets are dropped there, so a
        global basis can be given; a block without observations is NaN. Not available with
        ``full_covariance``; the log likelihood and mean coefficients, which differ per block,
        are NaN and None.

    Notes
    -----
    Cost: the n × n observation covariance is formed once and factored once per group of
    columns with the same valid observations (O(n³)); the posterior variances need one solve
    per target (O(n² m)). Thousands to a few tens of thousands of observations per estimate are
    practical.
    """
    y, nz, valid, single, groups = _columns(values, noise)
    n, k = y.shape
    names = covariance.names
    obs = _observation_atoms(coords, names, n)
    tgt = _grid_atoms(targets, names, target_weight)
    m = tgt.size
    terms, known = _parse_mean(mean)
    kernels = _kernels(coords, names)
    h, h_t, sd = _mean_design(terms, names, kernels, n, targets, target_weight)
    flat = np.isinf(sd)
    pf = int(flat.sum())
    local = bool(neighborhood)
    if local and full_covariance:
        msg = "full_covariance is not available with a neighborhood"
        raise ValueError(msg)
    blocks = (
        _blocks(neighborhood, names, targets, tgt, kernels, covariance)
        if local
        else [(np.arange(m), np.arange(n))]
    )

    g = np.full((m, k), np.nan)
    var = np.full((m, k), np.nan)
    used = np.zeros((m, k), dtype=int)
    ll = np.full(k, np.nan)
    beta = np.full((pf, k), np.nan)
    one_group = len(groups) == 1
    cov_out = np.empty((m, m) if one_group else (k, m, m)) if full_covariance else None
    w_out = np.zeros((m, n) if one_group else (k, m, n)) if weights else None
    off_out = np.full(m if one_group else (k, m), np.nan) if weights else None
    prior_all = np.full(m, np.nan)

    for t_idx, o_idx in blocks:
        whole = len(t_idx) == m and len(o_idx) == n
        obs_b = obs if whole else _take(obs, o_idx)
        tgt_b = tgt if whole else _take(tgt, t_idx)
        hb, hb_t = h[o_idx], h_t[t_idx]
        cols_b = (
            np.ones(len(sd), bool)
            if not local
            else np.any(hb != 0, axis=0) | np.any(hb_t != 0, axis=0)
        )
        fl = flat & cols_b
        pr = ~flat & cols_b
        # finite-prior mean terms are part of the covariance: Σ_p = diag(sd²)
        hp, hp_t = hb[:, pr] * sd[pr], hb_t[:, pr] * sd[pr]

        q0 = covariance.matrix(obs_b, obs_b, symmetric=True) if len(o_idx) else None
        b0 = covariance.matrix(tgt_b, obs_b) + hp_t @ hp.T if len(o_idx) else None
        if full_covariance:
            prior_cov = covariance.matrix(tgt_b, tgt_b, symmetric=True) + hp_t @ hp_t.T
            prior_var = np.diag(prior_cov).copy()
        else:
            prior_cov = None
            prior_var = covariance.diagonal(tgt_b) + np.einsum("mp,mp->m", hp_t, hp_t)
        prior_all[t_idx] = prior_var

        for cols in groups:
            idx = valid[o_idx, cols[0]]
            used[np.ix_(t_idx, cols)] = idx.sum()
            if not idx.any():
                continue  # no observations: NaN
            obs_rows = o_idx[idx]
            q = q0[np.ix_(idx, idx)] + np.diag(nz[obs_rows, cols[0]]) + hp[idx] @ hp[idx].T
            r = _solve(
                y[np.ix_(obs_rows, cols)],
                q,
                hb[idx][:, fl],
                known,
                b=b0[:, idx],
                hf_t=hb_t[:, fl],
                prior_var=prior_var,
                prior_cov=prior_cov,
                weights=weights,
            )
            g[np.ix_(t_idx, cols)] = r["g"]
            var[np.ix_(t_idx, cols)] = r["var"][:, None]
            if not local:
                ll[cols] = r["ll"]
                if pf:
                    beta[:, cols] = r["beta"]
            if one_group:
                if full_covariance:
                    cov_out[:] = r["cov"]
                if weights:
                    w_out[np.ix_(t_idx, obs_rows)] = r["w"]
                    off_out[t_idx] = r["offset"]
            else:
                for j in cols:
                    if full_covariance:
                        cov_out[j] = r["cov"]
                    if weights:
                        w_out[j][np.ix_(t_idx, obs_rows)] = r["w"]
                        off_out[j][t_idx] = r["offset"]
    prior_var = prior_all

    def shaped(a):
        a = a.reshape(*tgt.shape, k)
        return a[..., 0] if single else a

    is_constant = len(terms) == 1 and not terms[0].bases and pf == 1
    if known is not None:
        mean_out = known
    elif is_constant and not local:
        mean_out = float(beta[0, 0]) if single else beta[0]
    else:
        mean_out = float("nan")
    return Estimate(
        values=shaped(g),
        std=shaped(np.sqrt(np.maximum(var, 0.0))),
        prior_std=np.sqrt(np.maximum(prior_var, 0.0)).reshape(tgt.shape),
        mean=mean_out,
        log_likelihood=float(ll[0]) if single else ll,
        dims=names,
        mean_coefficients=None if pf == 0 or local else (beta[:, 0] if single else beta),
        covariance=cov_out,
        weights=w_out,
        offset=off_out,
        n_used=shaped(used),
    )
