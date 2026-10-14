"""fit_covariance: analytic gradients, consistency with log_likelihood, and recovery."""

from types import SimpleNamespace

import numpy as np
import pytest

import rebasis as rb
from rebasis import _fit

RNG = np.random.default_rng(11)
N = 120
X = RNG.uniform(0, 5, N)
T = RNG.uniform(0, 10, N)
Y = np.sin(1.3 * X) + 0.05 * T + RNG.normal(0, 0.1, N)
COORDS = {"x": X, "t": T}


def capture_objective(monkeypatch, **kwargs):
    """Run fit_covariance with the optimizer replaced, returning the objective and θ0."""
    seen = {}

    def fake_minimize(fun, x0, jac, method, bounds, options):
        seen["fun"], seen["x0"] = fun, np.asarray(x0)
        return SimpleNamespace(x=x0, fun=fun(x0)[0], success=True, message="", nit=0)

    monkeypatch.setattr(_fit.optimize, "minimize", fake_minimize)
    fit = rb.fit_covariance(Y, COORDS, **kwargs)
    return seen["fun"], seen["x0"], fit


def test_gradient_matches_finite_differences(monkeypatch):
    cov = rb.Covariance(
        1.0, x=rb.CovarianceDim("squared_exponential", 0.8), t=rb.CovarianceDim("matern32", 2.0)
    ) + rb.Covariance(
        0.3,
        x=rb.CovarianceDim("exponential", 3.0),
        t=rb.CovarianceDim("matern52", 6.0, period=10.0),
    )
    drift = rb.Space.shifted([5.0], [-5.0, 5.0], [-0.5, 0.5])
    mean = [rb.MeanTerm({}), rb.MeanTerm({"t": drift}, prior_std=0.2)]
    fun, theta0, fit = capture_objective(
        monkeypatch, covariance=cov, noise=0.01, mean=mean, nugget=0.02
    )
    assert set(fit.parameters) == {
        "0.variance",
        "0.length.x",
        "0.length.t",
        "1.variance",
        "1.length.x",
        "1.length.t",
        "mean.1.prior_std",
        "nugget",
    }
    theta = theta0 + RNG.normal(0, 0.2, len(theta0))  # away from the defaults
    _, g = fun(theta)
    for i in range(len(theta)):
        h = 1e-5
        e = np.zeros_like(theta)
        e[i] = h
        fd = (fun(theta + e)[0] - fun(theta - e)[0]) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-6), i


def test_known_mean_gradient(monkeypatch):
    cov = rb.Covariance(
        1.0, x=rb.CovarianceDim("matern32", 1.0), t=rb.CovarianceDim("exponential", 3.0)
    )
    fun, theta0, _ = capture_objective(
        monkeypatch, covariance=cov, noise=0.01, mean=0.3, fit_nugget=False
    )
    _, g = fun(theta0)
    for i in range(len(theta0)):
        e = np.zeros_like(theta0)
        e[i] = 1e-5
        fd = (fun(theta0 + e)[0] - fun(theta0 - e)[0]) / 2e-5
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_fitted_likelihood_is_log_likelihood():
    cov = rb.Covariance(
        1.0, x=rb.CovarianceDim("squared_exponential", 1.0), t=rb.CovarianceDim("matern32", 2.0)
    )
    fit = rb.fit_covariance(Y, COORDS, cov, noise=0.01)
    assert fit.success, fit.message
    ll = rb.log_likelihood(Y, COORDS, fit.covariance, noise=0.01 + fit.nugget, mean=fit.mean)
    assert fit.log_likelihood == pytest.approx(ll, rel=1e-10)
    # and it is a maximum: nudging a parameter lowers the likelihood
    for name in ("length.x", "length.t", "variance"):
        scaled = {k: v * (1.1 if k == name else 1.0) for k, v in fit.parameters.items()}
        c = rb.Covariance(
            scaled["variance"],
            x=rb.CovarianceDim("squared_exponential", scaled["length.x"]),
            t=rb.CovarianceDim("matern32", scaled["length.t"]),
        )
        assert rb.log_likelihood(Y, COORDS, c, noise=0.01 + fit.nugget) < fit.log_likelihood


def test_recovers_the_parameters_of_a_simulated_field():
    rng = np.random.default_rng(5)
    n = 600
    x, t = rng.uniform(0, 20, n), rng.uniform(0, 30, n)
    truth = rb.Covariance(
        2.0, x=rb.CovarianceDim("squared_exponential", 1.5), t=rb.CovarianceDim("matern32", 4.0)
    )
    obs = rb._estimate._observation_atoms({"x": x, "t": t}, truth.names, n)
    k = truth.matrix(obs, obs, symmetric=True) + 0.05 * np.eye(n)
    y = 1.0 + np.linalg.cholesky(k) @ rng.normal(size=n)
    start = rb.Covariance(
        0.5, x=rb.CovarianceDim("squared_exponential", 0.5), t=rb.CovarianceDim("matern32", 10.0)
    )
    fit = rb.fit_covariance(y, {"x": x, "t": t}, start, noise=0.05, fit_nugget=False)
    assert fit.success, fit.message
    p = fit.parameters
    assert p["length.x"] == pytest.approx(1.5, rel=0.25)
    assert p["length.t"] == pytest.approx(4.0, rel=0.35)
    assert p["variance"] == pytest.approx(2.0, rel=0.6)


def test_estimate_with_the_fit_and_options():
    cov = rb.Covariance(
        1.0, x=rb.CovarianceDim("squared_exponential", 1.0), t=rb.CovarianceDim("matern32", 2.0)
    )
    fit = rb.fit_covariance(
        Y,
        COORDS,
        cov,
        noise=0.01,
        fixed={"length.t"},
        bounds={"length.x": (0.2, 5.0)},
        subsample=80,
    )
    assert fit.parameters["length.t"] == 2.0 and fit.n_used == 80
    assert 0.2 <= fit.parameters["length.x"] <= 5.0
    targets = {"x": rb.Space.boxes([0.0, 2.5, 5.0]), "t": rb.Space.boxes([0.0, 10.0])}
    a = fit.estimate(Y, COORDS, targets, noise=0.01)
    b = rb.optimal_estimate(
        Y, COORDS, targets, fit.covariance, noise=0.01 + fit.nugget, mean=fit.mean
    )
    np.testing.assert_allclose(a.values, b.values)
    with pytest.raises(ValueError, match="unknown parameters"):
        rb.fit_covariance(Y, COORDS, cov, noise=0.01, fixed={"length.z"})
