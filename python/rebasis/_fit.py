"""Fitting covariance parameters by maximum (restricted) likelihood, with analytic gradients."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, optimize

from . import _rebasis
from ._estimate import (
    Covariance,
    CovarianceDim,
    CovarianceSum,
    Estimate,
    MeanTerm,
    _kernels,
    _mean_design,
    _observation_atoms,
    _parse_mean,
    optimal_estimate,
)
from ._space import Space


@dataclass
class CovarianceFit:
    """The result of :func:`fit_covariance`."""

    covariance: Covariance | CovarianceSum
    """The fitted covariance."""
    mean: object
    """The mean specification with fitted prior standard deviations, for
    :func:`optimal_estimate`."""
    nugget: float
    """The fitted variance added to the given noise (0 if not fitted)."""
    log_likelihood: float
    """The maximized log likelihood (of the data used, e.g. the subsample)."""
    parameters: dict[str, float]
    """Every parameter by name, in natural units."""
    success: bool
    message: str
    n_iterations: int
    n_used: int
    """The number of observations used (after dropping invalid ones and subsampling)."""
    history: list[float] = field(default_factory=list)

    def estimate(self, values, coords, targets, *, noise, **kwargs) -> Estimate:
        """:func:`optimal_estimate` with the fitted covariance, mean and nugget."""
        return optimal_estimate(
            values,
            coords,
            targets,
            self.covariance,
            noise=np.asarray(noise, dtype=np.float64) + self.nugget,
            mean=self.mean,
            **kwargs,
        )


@dataclass
class _Param:
    name: str
    value: float  # natural units, initial
    lo: float
    hi: float


def _components(cov) -> list[Covariance]:
    return list(cov.components) if isinstance(cov, CovarianceSum) else [cov]


def fit_covariance(
    values,
    coords: Mapping,
    covariance: Covariance | CovarianceSum,
    *,
    noise,
    mean="estimate",
    nugget: float | None = None,
    fit_nugget: bool = True,
    fixed: set[str] | frozenset[str] = frozenset(),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    subsample: int | None = None,
    seed: int = 0,
    maxiter: int = 200,
) -> CovarianceFit:
    """Fit the covariance (and the mean's prior widths and a nugget) to data by maximizing the
    restricted log likelihood, with analytic gradients.

    The parameters, named as in :attr:`CovarianceFit.parameters`, are fitted in log space:

    - ``variance`` and ``length.<dim>`` of the covariance (``<i>.variance`` and
      ``<i>.length.<dim>`` for component i of a :class:`CovarianceSum`), starting from the
      values in ``covariance``; families and periods are kept;
    - ``mean.<i>.prior_std`` for every :class:`MeanTerm` with a finite prior: a common factor on
      its prior standard deviations, reported as the fitted value (the largest when they vary);
    - ``nugget``: a variance added to ``noise`` for variability the covariance does not resolve,
      starting from ``nugget`` or 1 % of the data variance (with ``fit_nugget``).

    Parameters
    ----------
    values, coords, noise, mean
        As in :func:`optimal_estimate` (one column of values; invalid values are dropped).
    fixed
        Names of parameters to keep at their initial values.
    bounds
        Name -> (lo, hi) in natural units. Defaults: variances 1e-4 to 1e4 times their initial
        values, lengths 1e-2 to 1e2 times, prior widths 1e-4 to 1e4 times, the nugget 1e-10 to
        10 times the data variance.
    subsample
        Fit on at most this many randomly chosen observations: each likelihood evaluation costs
        O(n³), and a few thousand observations usually determine the parameters well.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        msg = "fit_covariance takes one column of values"
        raise ValueError(msg)
    n_all = len(y)
    nz = np.broadcast_to(np.asarray(noise, dtype=np.float64), (n_all,))
    keep = np.isfinite(y) & np.isfinite(nz)
    if subsample is not None and keep.sum() > subsample:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(np.flatnonzero(keep), subsample, replace=False)
        keep = np.zeros(n_all, bool)
        keep[chosen] = True
    idx = np.flatnonzero(keep)
    names = covariance.names
    sub = {}
    for name in names:
        c = coords[name]
        sub[name] = c.take(idx) if isinstance(c, Space) else np.asarray(c, float).ravel()[idx]
    y, nz = y[idx], np.array(nz[idx])
    n = len(y)
    obs = _observation_atoms(sub, names, n)
    kernels = _kernels(sub, names)

    terms, known = _parse_mean(mean)
    term_h = [_mean_design([t], names, kernels, n, None, None) for t in terms]
    h_flat = [h for (h, _, sd) in term_h if np.all(np.isinf(sd))]
    finite = [
        (i, h, sd) for i, (h, _, sd) in enumerate(term_h) if np.all(np.isfinite(sd)) and h.size
    ]
    if any(np.any(np.isinf(sd)) and np.any(np.isfinite(sd)) for (_, _, sd) in term_h):
        msg = "a MeanTerm's prior_std must be all finite or all infinite"
        raise ValueError(msg)
    hf = np.hstack(h_flat) if h_flat else np.zeros((n, 0))

    var_y = float(np.var(y)) if n > 1 else 1.0
    bounds = dict(bounds or {})
    comps = _components(covariance)
    multi = len(comps) > 1
    params: list[_Param] = []
    for ci, comp in enumerate(comps):
        pre = f"{ci}." if multi else ""
        params.append(
            _Param(f"{pre}variance", comp.variance, 1e-4 * comp.variance, 1e4 * comp.variance)
        )
        for dim, cd in comp.dims.items():
            params.append(
                _Param(f"{pre}length.{dim}", cd.length, 1e-2 * cd.length, 1e2 * cd.length)
            )
    for i, _, sd in finite:
        s0 = float(np.max(sd)) if np.max(sd) > 0 else 1.0
        params.append(_Param(f"mean.{i}.prior_std", s0, 1e-4 * s0, 1e4 * s0))
    if fit_nugget:
        t0 = nugget if nugget is not None else 0.01 * var_y
        params.append(_Param("nugget", t0, 1e-10 * var_y, 10 * var_y))
    by_name = {p.name: p for p in params}
    unknown = (set(bounds) | set(fixed)) - set(by_name)
    if unknown:
        msg = f"unknown parameters {sorted(unknown)}; available: {sorted(by_name)}"
        raise ValueError(msg)
    for name, (lo, hi) in bounds.items():
        if not (0 < lo <= by_name[name].value <= hi):
            msg = f"bounds for {name!r} must be positive and contain its initial value"
            raise ValueError(msg)
        by_name[name].lo, by_name[name].hi = lo, hi
    free = [p for p in params if p.name not in fixed]
    fixed_nugget = 0.0 if fit_nugget else (nugget or 0.0)

    def unpack(theta):
        vals = {p.name: p.value for p in params}
        vals.update({p.name: float(np.exp(t)) for p, t in zip(free, theta, strict=True)})
        return vals

    def model(vals):
        out = []
        for ci, comp in enumerate(comps):
            pre = f"{ci}." if multi else ""
            dims = {
                dim: CovarianceDim(cd.family, vals[f"{pre}length.{dim}"], cd.period)
                for dim, cd in comp.dims.items()
            }
            out.append(Covariance(vals[f"{pre}variance"], **dims))
        return out

    one = np.ones(n)
    history: list[float] = []

    def objective(theta):
        vals = unpack(theta)
        comps_v = model(vals)
        mats = []
        q = np.diag(nz + (vals["nugget"] if fit_nugget else fixed_nugget))
        for c in comps_v:
            k, g = _rebasis.covariance_length_derivatives(
                c.variance, c._spec(), obs.spaces, obs.weights
            )
            mats.append((k, g))
            q += k
        prior_terms = []
        for i, h, sd in finite:
            s = vals[f"mean.{i}.prior_std"]
            scale = s / float(np.max(sd)) if np.max(sd) > 0 else s
            hs = h * (sd * scale)
            prior_terms.append(hs)
            q += hs @ hs.T
        try:
            cf = linalg.cho_factor(q, lower=True, check_finite=False)
        except linalg.LinAlgError:
            return 1e30, np.zeros(len(theta))
        qi = linalg.cho_solve(cf, np.eye(n), check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
        if hf.shape[1]:
            qi_h = qi @ hf
            s_mat = hf.T @ qi_h
            try:
                s_cf = linalg.cho_factor(s_mat, lower=True, check_finite=False)
            except linalg.LinAlgError:
                return 1e30, np.zeros(len(theta))
            pmat = qi - qi_h @ linalg.cho_solve(s_cf, qi_h.T, check_finite=False)
            alpha = pmat @ y
            p_f = hf.shape[1]
            ll = (
                -0.5 * y @ alpha
                - 0.5 * logdet
                - np.sum(np.log(np.diag(s_cf[0])))
                + 0.5 * np.linalg.slogdet(hf.T @ hf)[1]
                - 0.5 * (n - p_f) * math.log(2 * math.pi)
            )
        else:
            pmat = qi
            alpha = qi @ (y - (known or 0.0) * one)
            ll = (
                -0.5 * (y - (known or 0.0)) @ alpha - 0.5 * logdet - 0.5 * n * math.log(2 * math.pi)
            )

        # ∂ℓ/∂θ = ½ (αᵀ ∂Q α − tr(P ∂Q)) for each log parameter
        def half(dq):
            return 0.5 * (alpha @ dq @ alpha - np.sum(pmat * dq))

        grads = {}
        for ci, (k, g) in enumerate(mats):
            pre = f"{ci}." if multi else ""
            grads[f"{pre}variance"] = half(k)
            for (dim, _), gd in zip(comps[ci].dims.items(), g, strict=True):
                grads[f"{pre}length.{dim}"] = half(gd)
        for (i, _, _), hs in zip(finite, prior_terms, strict=True):
            # Q contains s² H D Hᵀ: ∂/∂log s = 2 H_s H_sᵀ
            hta = hs.T @ alpha
            grads[f"mean.{i}.prior_std"] = 0.5 * (2 * hta @ hta - 2 * np.sum((pmat @ hs) * hs))
        if fit_nugget:
            tau = vals["nugget"]
            grads["nugget"] = 0.5 * tau * (alpha @ alpha - np.trace(pmat))
        history.append(float(ll))
        return -float(ll), -np.array([grads[p.name] for p in free])

    theta0 = np.log([p.value for p in free])
    res = optimize.minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(math.log(p.lo), math.log(p.hi)) for p in free],
        options={"maxiter": maxiter},
    )
    vals = unpack(res.x)
    comps_v = model(vals)
    cov_out = comps_v[0] if not multi else CovarianceSum(*comps_v)
    if known is not None:
        mean_out = known
    else:
        new_terms = []
        for i, t in enumerate(terms):
            sd = np.asarray(t.prior_std, dtype=np.float64)
            if np.all(np.isfinite(sd)) and f"mean.{i}.prior_std" in vals:
                s0 = float(np.max(sd)) if np.max(sd) > 0 else 1.0
                new = sd * (vals[f"mean.{i}.prior_std"] / s0)
                new_terms.append(MeanTerm(t.bases, float(new) if new.ndim == 0 else new))
            else:
                new_terms.append(t)
        mean_out = new_terms
    return CovarianceFit(
        covariance=cov_out,
        mean=mean_out,
        nugget=vals["nugget"] if fit_nugget else fixed_nugget,
        log_likelihood=-float(res.fun),
        parameters=vals,
        success=bool(res.success),
        message=str(res.message),
        n_iterations=int(res.nit),
        n_used=n,
        history=history,
    )
