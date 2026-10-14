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
    """

    family: str
    length: float
    period: float | None = None

    def __post_init__(self):
        if self.family not in FAMILIES and self.family not in ("gaussian", "matern12"):
            msg = f"family must be one of {FAMILIES}, got {self.family!r}"
            raise ValueError(msg)

    def _spec(self):
        period = None if self.period is None else float(self.period)
        return (self.family, float(self.length), period)


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


@dataclass
class Estimate:
    """The result of :func:`optimal_estimate`. Arrays have the shape of the target grid."""

    values: np.ndarray
    """The estimates ĝ."""
    std: np.ndarray
    """Their posterior standard deviations (including the uncertainty of an estimated mean)."""
    prior_std: np.ndarray
    """The prior standard deviations of the targets, for comparison with ``std``."""
    mean: float
    """The (estimated or given) constant mean β (NaN for a mean basis; see
    ``mean_coefficients``)."""
    log_likelihood: float
    """log p(y | covariance), restricted (REML, in Harville's basis-independent form) when the
    mean is estimated; maximize it over the covariance parameters to fit them."""
    dims: tuple[str, ...]
    mean_coefficients: np.ndarray | None = None
    """The estimated coefficients of a mean basis (``mean`` given as spaces)."""
    covariance: np.ndarray | None = None
    """The posterior covariance of the flattened targets, if requested."""
    weights: np.ndarray | None = None
    """W, shape (n_targets, n_observations): ĝ = W y (+ a constant for a known mean). Row m is
    target m's averaging kernel over the observations; with an estimated mean every row sums
    to 1."""
    offset: np.ndarray | None = None

    def apply(self, values) -> np.ndarray:
        """The same estimator applied to other values at the same observations, e.g. a model
        field sampled like the data: comparing that with the model's true averages measures
        the sampling error."""
        if self.weights is None:
            msg = "the estimate was computed with weights=False"
            raise ValueError(msg)
        out = self.weights @ np.asarray(values, dtype=np.float64) + self.offset
        return out.reshape(self.values.shape)

    def weight_histogram(self, coord, edges) -> np.ndarray:
        """Σ of each target's weights over the observations whose coordinate ``coord`` (one value
        per observation) falls in each bin of ``edges``: how the estimate draws on, e.g., each
        day of the month. Shape: target grid + (len(edges) − 1,)."""
        if self.weights is None:
            msg = "the estimate was computed with weights=False"
            raise ValueError(msg)
        coord = np.asarray(coord, dtype=np.float64)
        b = np.digitize(coord, edges) - 1
        nb = len(edges) - 1
        out = np.zeros((self.weights.shape[0], nb))
        for k in range(nb):
            out[:, k] = self.weights[:, b == k].sum(axis=1)
        return out.reshape(*self.values.shape, nb)


def _mean_design(
    mean_basis: Mapping, names, obs: _Atoms, tgt: _Atoms | None, coords, targets, target_weight
):
    """H for the observations (n × P, a row-wise Kronecker product of per-dimension values) and
    for the target grid (m × P, a Kronecker product), for a mean Σ_p β_p Π_d h_pd(x_d)."""
    unknown = set(mean_basis) - set(names)
    if unknown:
        msg = f"mean basis for unknown dimensions {sorted(unknown)}"
        raise ValueError(msg)
    target_weight = dict(target_weight or {})
    h_obs = np.ones((obs.size, 1))
    h_tgt = np.ones((1, 1)) if tgt is not None else None
    for name in names:
        basis = mean_basis.get(name)
        if basis is None:
            continue
        c = coords[name]
        kern = c if isinstance(c, Space) else Space.deltas(np.asarray(c, dtype=np.float64).ravel())
        ho = measure(kern, basis).toarray()  # (n, P_d)
        h_obs = (h_obs[:, :, None] * ho[:, None, :]).reshape(obs.size, -1)
    if tgt is not None:
        for name in names:
            basis = mean_basis.get(name)
            if basis is None:
                ht = np.ones((len(targets[name]), 1))
            else:
                ht = measure(targets[name], basis, weight=target_weight.get(name)).toarray()
            h_tgt = np.kron(h_tgt, ht)
    return h_obs, h_tgt


class _Mean:
    """The generalized-least-squares mean: β = S⁻¹ Hᵀ Q⁻¹ y, S = Hᵀ Q⁻¹ H."""

    def __init__(self, cf, h, y):
        self.h = h
        self.qi_h = linalg.cho_solve(cf, h, check_finite=False)  # (n, P)
        s = h.T @ self.qi_h
        try:
            self.s_cf = linalg.cho_factor(s, lower=True, check_finite=False)
        except linalg.LinAlgError as e:
            msg = (
                "the observations do not determine every mean coefficient: use fewer mean basis "
                "functions or ones covered by the data"
            )
            raise _rebasis.SingularError(msg) from e
        self.beta = linalg.cho_solve(self.s_cf, self.qi_h.T @ y, check_finite=False)
        self.logdet_s = 2.0 * np.sum(np.log(np.diag(self.s_cf[0])))
        # ½ log|HᵀH| makes the restricted likelihood independent of how the mean basis is
        # parametrized (Harville 1974), so likelihoods compare across mean bases
        self.logdet_hh = np.linalg.slogdet(h.T @ h)[1]


def _factor(q: np.ndarray):
    try:
        return linalg.cho_factor(q, lower=True, check_finite=False)
    except linalg.LinAlgError as e:
        msg = (
            "the observation covariance is not positive definite; observations at identical "
            "coordinates need noise > 0"
        )
        raise _rebasis.SingularError(msg) from e


def _log_likelihood(cf, r, alpha, mean: _Mean | None) -> float:
    n = len(r)
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    ll = -0.5 * float(r @ alpha) - 0.5 * logdet
    if mean is None:
        return ll - 0.5 * n * math.log(2 * math.pi)
    p = mean.h.shape[1]
    return ll - 0.5 * mean.logdet_s + 0.5 * mean.logdet_hh - 0.5 * (n - p) * math.log(2 * math.pi)


def _parse_mean(mean):
    """None (known value) / {} (constant) / {dim: Space}, and the known value."""
    if isinstance(mean, str):
        if mean != "estimate":
            msg = f"mean must be 'estimate', a number, or a dict of spaces; got {mean!r}"
            raise ValueError(msg)
        return {}, None
    if isinstance(mean, Mapping):
        return dict(mean), None
    return None, float(mean)


def log_likelihood(values, coords: Mapping, covariance: Covariance, *, noise, mean="estimate"):
    """log p(y | covariance) without computing any targets; see :attr:`Estimate.log_likelihood`.
    Cheap enough to maximize over the covariance parameters with ``scipy.optimize``."""
    y = np.asarray(values, dtype=np.float64).ravel()
    obs = _observation_atoms(coords, covariance.names, len(y))
    q = covariance.matrix(obs, obs, symmetric=True)
    q[np.diag_indices_from(q)] += np.broadcast_to(np.asarray(noise, dtype=np.float64), y.shape)
    cf = _factor(q)
    basis, known = _parse_mean(mean)
    if basis is None:
        gls, r = None, y - known
    else:
        h, _ = _mean_design(basis, covariance.names, obs, None, coords, None, None)
        gls = _Mean(cf, h, y)
        r = y - h @ gls.beta
    return _log_likelihood(cf, r, linalg.cho_solve(cf, r, check_finite=False), gls)


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
) -> Estimate:
    """Optimal (kriging) estimates of grid averages from irregular observations.

    Parameters
    ----------
    values
        The n observations.
    coords
        For each dimension of ``covariance``: the n observation coordinates (point
        observations), or a :class:`Space` of n elements (each observation an average over its
        element, e.g. a footprint or a time window).
    targets
        For each dimension, a :class:`Space` of target elements; the targets are their tensor
        product (e.g. ``{"lat": Space.boxes(lat_edges), "lon": Space.boxes([0, 360]),
        "time": Space.boxes(month_edges)}`` for monthly zonal means).
    covariance
        The prior :class:`Covariance`.
    noise
        Observation error variances (scalar or one per observation).
    mean
        ``"estimate"`` (default): an unknown constant, estimated by generalized least squares
        (ordinary kriging); every row of the weights then sums to 1, so a constant field is
        reproduced exactly however it is sampled. A dict ``{dim: Space}``: a mean that is a
        tensor product of these bases (constant along the other dimensions), with estimated
        coefficients (universal kriging), e.g. ``{"lat": Space.bsplines_clamped(...),
        "time": Space.hats([t0, t1])}`` for a latitude structure drifting linearly in time;
        any field in that span is reproduced exactly. Or a known value.
    target_weight
        Per dimension, a weight for the target averages, e.g. ``{"lat": "cos_lat_deg"}``.
    weights, full_covariance
        Also return the weights (n_targets × n) and the posterior covariance of the targets.

    Notes
    -----
    Cost: the n × n observation covariance is formed and factored (O(n³)), and the posterior
    variances need one solve per target (O(n² m)); thousands to a few tens of thousands of
    observations per estimate are practical.
    """
    y = np.asarray(values, dtype=np.float64).ravel()
    n = len(y)
    names = covariance.names
    obs = _observation_atoms(coords, names, n)
    tgt = _grid_atoms(targets, names, target_weight)
    noise = np.broadcast_to(np.asarray(noise, dtype=np.float64), (n,))
    if np.any(noise < 0) or not np.all(np.isfinite(noise)):
        msg = "noise variances must be finite and >= 0"
        raise ValueError(msg)

    q = covariance.matrix(obs, obs, symmetric=True)
    q[np.diag_indices_from(q)] += noise
    cf = _factor(q)
    b = covariance.matrix(tgt, obs)  # (m, n)

    basis, known = _parse_mean(mean)
    if basis is None:
        gls, h_tgt = None, None
        mu_obs, mu_tgt = np.full(n, known), np.full(tgt.size, known)
    else:
        h_obs, h_tgt = _mean_design(basis, names, obs, tgt, coords, targets, target_weight)
        gls = _Mean(cf, h_obs, y)
        mu_obs, mu_tgt = h_obs @ gls.beta, h_tgt @ gls.beta
    r = y - mu_obs
    alpha = linalg.cho_solve(cf, r, check_finite=False)
    g = mu_tgt + b @ alpha

    x = linalg.cho_solve(cf, b.T, check_finite=False)  # Q⁻¹ Bᵀ, (n, m)
    if full_covariance:
        prior = covariance.matrix(tgt, tgt, symmetric=True)
        prior_var = np.diag(prior).copy()
    else:
        prior = None
        prior_var = covariance.diagonal(tgt)
    var = prior_var - np.einsum("mn,nm->m", b, x)
    if gls is not None:
        # the mean's uncertainty: R S⁻¹ Rᵀ, R = H_t − B Q⁻¹ H
        rmat = h_tgt - b @ gls.qi_h
        s_inv_rt = linalg.cho_solve(gls.s_cf, rmat.T, check_finite=False)
        var += np.einsum("mp,pm->m", rmat, s_inv_rt)

    est = Estimate(
        values=g.reshape(tgt.shape),
        std=np.sqrt(np.maximum(var, 0.0)).reshape(tgt.shape),
        prior_std=np.sqrt(prior_var).reshape(tgt.shape),
        mean=float(gls.beta[0])
        if gls is not None and len(gls.beta) == 1
        else (known if gls is None else float("nan")),
        log_likelihood=_log_likelihood(cf, r, alpha, gls),
        dims=names,
        mean_coefficients=None if gls is None else gls.beta,
    )
    if full_covariance:
        post = prior - b @ x
        if gls is not None:
            post += rmat @ s_inv_rt
        est.covariance = post
    if weights:
        w = x.T.copy()
        if gls is not None:
            # W = B Q⁻¹ + R S⁻¹ Hᵀ Q⁻¹: unbiased for every field in the mean's span
            w += s_inv_rt.T @ gls.qi_h.T
            est.offset = np.zeros(len(g))
        else:
            est.offset = known * (1.0 - w.sum(axis=1))
        est.weights = w
    return est
