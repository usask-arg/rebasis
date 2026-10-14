"""Optimal estimation against an independent numpy implementation and known limits."""

import itertools

import numpy as np
import pytest

import rebasis as rb

RNG = np.random.default_rng(42)
N = 80
OBS_X = RNG.uniform(0, 5, N)
OBS_T = RNG.uniform(0, 10, N)
COV = rb.Covariance(
    2.0,
    x=rb.CovarianceDim("squared_exponential", 0.7),
    t=rb.CovarianceDim("matern32", 1.5),
)


def field(x, t):
    return 1.0 + np.sin(1.3 * x) * np.cos(0.4 * t) + 0.1 * t


Y = field(OBS_X, OBS_T) + RNG.normal(0, 0.05, N)


def kernel(x1, t1, x2, t2):
    dx = x1[:, None] - x2[None, :]
    u = np.sqrt(3) * np.abs(t1[:, None] - t2[None, :]) / 1.5
    return 2.0 * np.exp(-(dx**2) / (2 * 0.7**2)) * (1 + u) * np.exp(-u)


def reference_kriging(xs, ts, noise):
    """Ordinary kriging at points, written out independently of rebasis."""
    q = kernel(OBS_X, OBS_T, OBS_X, OBS_T) + noise * np.eye(N)
    k = kernel(xs, ts, OBS_X, OBS_T)
    qi = np.linalg.inv(q)
    one = np.ones(N)
    s = one @ qi @ one
    beta = one @ qi @ Y / s
    g = beta + k @ qi @ (Y - beta)
    r = 1 - k @ qi @ one
    var = 2.0 - np.einsum("ij,jk,ik->i", k, qi, k) + r**2 / s
    logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(q))))
    res = Y - beta
    ll = (
        -0.5 * res @ qi @ res
        - 0.5 * logdet
        - 0.5 * np.log(s)
        + 0.5 * np.log(N)
        - 0.5 * (N - 1) * np.log(2 * np.pi)
    )
    return g, np.sqrt(var), beta, ll


def test_point_targets_match_independent_kriging():
    gx, gt = np.linspace(0.5, 4.5, 5), np.linspace(1, 9, 4)
    est = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.deltas(gx), "t": rb.Space.deltas(gt)},
        COV,
        noise=0.05**2,
        full_covariance=True,
    )
    xx, tt = np.meshgrid(gx, gt, indexing="ij")
    g, std, beta, ll = reference_kriging(xx.ravel(), tt.ravel(), 0.05**2)
    np.testing.assert_allclose(est.values.ravel(), g, rtol=1e-9)
    np.testing.assert_allclose(est.std.ravel(), std, rtol=1e-7)
    assert est.mean == pytest.approx(beta, rel=1e-10)
    assert est.log_likelihood == pytest.approx(ll, rel=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(est.covariance)), std, rtol=1e-7)
    ll2 = rb.log_likelihood(Y, {"x": OBS_X, "t": OBS_T}, COV, noise=0.05**2)
    assert ll2 == pytest.approx(est.log_likelihood, rel=1e-12)


def test_box_target_is_the_limit_of_point_targets():
    # a box average in x at a point in t, against the trapezoid average of point targets
    lo, hi = 1.2, 2.9
    box = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.boxes([lo, hi]), "t": rb.Space.deltas([4.0])},
        COV,
        noise=0.05**2,
    )
    xs = np.linspace(lo, hi, 4001)
    pts = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.deltas(xs), "t": rb.Space.deltas([4.0])},
        COV,
        noise=0.05**2,
    )
    avg = np.trapezoid(pts.values[:, 0], xs) / (hi - lo)
    assert box.values[0, 0] == pytest.approx(avg, abs=1e-7)
    # the prior variance of the box mean is the double average of the point covariance
    k = kernel(xs, np.full_like(xs, 4.0), xs, np.full_like(xs, 4.0))
    double = np.trapezoid(np.trapezoid(k, xs, axis=1), xs) / (hi - lo) ** 2
    assert box.prior_std[0, 0] ** 2 == pytest.approx(double, rel=1e-6)


def test_weights_sum_to_one_and_reproduce_constants():
    est = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.boxes([0, 1, 2.5, 5]), "t": rb.Space.boxes([0, 5, 10])},
        COV,
        noise=0.05**2,
    )
    np.testing.assert_allclose(est.weights.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(est.apply(Y), est.values, rtol=1e-10)
    np.testing.assert_allclose(est.apply(np.full(N, 3.7)), 3.7, rtol=1e-10)


def test_known_mean():
    est = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.deltas([2.0]), "t": rb.Space.deltas([5.0])},
        COV,
        noise=0.05**2,
        mean=1.0,
    )
    q = kernel(OBS_X, OBS_T, OBS_X, OBS_T) + 0.05**2 * np.eye(N)
    k = kernel(np.array([2.0]), np.array([5.0]), OBS_X, OBS_T)
    expected = 1.0 + k @ np.linalg.solve(q, Y - 1.0)
    assert est.values.ravel()[0] == pytest.approx(expected[0], rel=1e-10)
    assert est.mean == 1.0
    np.testing.assert_allclose(est.apply(Y), est.values, rtol=1e-10)


def test_periodic_longitude_and_area_weighted_zonal_means():
    rng = np.random.default_rng(7)
    lat = rng.uniform(-60, 60, 300)
    lon = rng.uniform(0, 360, 300)
    y = np.cos(np.radians(lat)) * (1 + 0.2 * np.sin(np.radians(2 * lon)))
    cov = rb.Covariance(
        0.5,
        lat=rb.CovarianceDim("squared_exponential", 8.0),
        lon=rb.CovarianceDim("exponential", 60.0, period=360.0),
    )
    targets = {"lat": rb.Space.boxes([-40, -20, 0, 20, 40]), "lon": rb.Space.boxes([0, 360])}
    a = rb.optimal_estimate(
        y, {"lat": lat, "lon": lon}, targets, cov, noise=1e-4, target_weight={"lat": "cos_lat_deg"}
    )
    b = rb.optimal_estimate(
        y,
        {"lat": lat, "lon": lon + 360.0},
        targets,
        cov,
        noise=1e-4,
        target_weight={"lat": "cos_lat_deg"},
    )
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)
    # truth: the area-weighted mean of cos(lat) over each band (the wave averages out)
    edges = np.array([-40, -20, 0, 20, 40.0])
    r = np.radians(edges)
    truth = [
        (0.5 * (r1 - r0) + 0.25 * (np.sin(2 * r1) - np.sin(2 * r0))) / (np.sin(r1) - np.sin(r0))
        for r0, r1 in itertools.pairwise(r)
    ]
    err = np.abs(a.values[:, 0] - truth)
    assert np.all(err < 3 * a.std[:, 0] + 1e-4), (err, a.std[:, 0])
    assert np.all(a.std < 0.2 * a.prior_std)


def test_time_window_observations():
    # observations that are themselves averages over a time window
    t0 = np.sort(RNG.uniform(0, 9, 30))
    windows = rb.Space.rectangles(t0 + 0.5, width=1.0)
    est = rb.optimal_estimate(
        RNG.normal(size=30),
        {"t": windows},
        {"t": rb.Space.boxes([0, 10])},
        rb.Covariance(1.0, t=rb.CovarianceDim("matern52", 2.0)),
        noise=0.01,
    )
    assert np.isfinite(est.values).all() and 0 < est.std[0] < est.prior_std[0]


def test_errors():
    with pytest.raises(rb.SingularError, match="noise"):
        rb.optimal_estimate(
            [1.0, 2.0],
            {"x": [0.5, 0.5]},
            {"x": rb.Space.boxes([0, 1])},
            rb.Covariance(1.0, x=rb.CovarianceDim("exponential", 1.0)),
            noise=0.0,
        )
    with pytest.raises(ValueError, match="dimensions"):
        rb.optimal_estimate(Y, {"x": OBS_X}, {"x": rb.Space.boxes([0, 1])}, COV, noise=0.1)
    with pytest.raises(ValueError, match="family"):
        rb.CovarianceDim("bogus", 1.0)


def test_universal_kriging_matches_independent_implementation():
    # mean linear in x: hats on [0, 5] span {1, x}
    gx = np.array([0.7, 2.2, 4.1])
    est = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.deltas(gx), "t": rb.Space.deltas([3.0])},
        COV,
        noise=0.05**2,
        mean={"x": rb.Space.hats([0.0, 5.0])},
    )
    q = kernel(OBS_X, OBS_T, OBS_X, OBS_T) + 0.05**2 * np.eye(N)
    k = kernel(gx, np.full(3, 3.0), OBS_X, OBS_T)
    h = np.column_stack([np.ones(N), OBS_X])
    ht = np.column_stack([np.ones(3), gx])
    qi = np.linalg.inv(q)
    s = h.T @ qi @ h
    beta = np.linalg.solve(s, h.T @ qi @ Y)
    g = ht @ beta + k @ qi @ (Y - h @ beta)
    r = ht - k @ qi @ h
    var = (
        2.0 - np.einsum("ij,jk,ik->i", k, qi, k) + np.einsum("ip,pq,iq->i", r, np.linalg.inv(s), r)
    )
    np.testing.assert_allclose(est.values.ravel(), g, rtol=1e-9)
    np.testing.assert_allclose(est.std.ravel(), np.sqrt(var), rtol=1e-7)
    res = Y - h @ beta
    logdet_q = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(q))))
    logdet_s = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(s))))
    logdet_hh = np.linalg.slogdet(h.T @ h)[1]  # with the {1, x} basis
    ll = (
        -0.5 * res @ qi @ res
        - 0.5 * logdet_q
        - 0.5 * logdet_s
        + 0.5 * logdet_hh
        - 0.5 * (N - 2) * np.log(2 * np.pi)
    )
    assert est.log_likelihood == pytest.approx(ll, rel=1e-10)


def test_fields_in_the_mean_span_are_reproduced_exactly():
    # f = a(x) + c t with a(x) a cubic spline: in the span of splines(x) ⊗ hats(t)
    xb = rb.Space.bsplines_clamped(np.linspace(0, 5, 6), 3)
    tb = rb.Space.hats([0.0, 10.0])
    coef = RNG.normal(size=len(xb))
    a = lambda x: xb.evaluate(x) @ coef  # noqa: E731
    y = a(OBS_X) + 0.3 * OBS_T
    targets = {"x": rb.Space.boxes([0.0, 1.5, 4.0]), "t": rb.Space.boxes([2.0, 9.0])}
    est = rb.optimal_estimate(
        y, {"x": OBS_X, "t": OBS_T}, targets, COV, noise=1e-6, mean={"x": xb, "t": tb}
    )
    xs = np.linspace(0, 4, 40001)
    truth = [
        np.trapezoid(a(xs[(xs >= lo) & (xs <= hi)]), xs[(xs >= lo) & (xs <= hi)]) / (hi - lo)
        + 0.3 * 5.5
        for lo, hi in [(0.0, 1.5), (1.5, 4.0)]
    ]
    np.testing.assert_allclose(est.values[:, 0], truth, atol=1e-7)
    # the weights are unbiased for any such field, whatever its coefficients
    y2 = a(OBS_X) * 2 - 0.1 * OBS_T
    truth2 = 2 * (np.array(truth) - 0.3 * 5.5) - 0.1 * 5.5
    np.testing.assert_allclose(est.apply(y2)[:, 0], truth2, atol=1e-7)


def test_covariance_sums_add_matrices():
    short = rb.Covariance(
        1.0, x=rb.CovarianceDim("exponential", 0.5), t=rb.CovarianceDim("matern32", 1.0)
    )
    long = rb.Covariance(
        0.5, x=rb.CovarianceDim("squared_exponential", 3.0), t=rb.CovarianceDim("matern52", 8.0)
    )
    total = short + long
    targets = {"x": rb.Space.boxes([0.0, 2.5, 5.0]), "t": rb.Space.boxes([0.0, 10.0])}
    a = rb.optimal_estimate(
        Y, {"x": OBS_X, "t": OBS_T}, targets, total, noise=0.01, full_covariance=True
    )

    # reference: point covariances of the sum, written out
    def k(x1, t1, x2, t2):
        dx = np.abs(x1[:, None] - x2[None, :])
        dt = np.abs(t1[:, None] - t2[None, :])
        u1, u2 = np.sqrt(3) * dt / 1.0, np.sqrt(5) * dt / 8.0
        return np.exp(-dx / 0.5) * (1 + u1) * np.exp(-u1) + 0.5 * np.exp(-(dx**2) / 18.0) * (
            1 + u2 + u2**2 / 3
        ) * np.exp(-u2)

    q = k(OBS_X, OBS_T, OBS_X, OBS_T) + 0.01 * np.eye(N)
    # one point target, to compare with closed forms
    p = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.deltas([1.3]), "t": rb.Space.deltas([4.0])},
        total,
        noise=0.01,
    )
    qi = np.linalg.inv(q)
    one = np.ones(N)
    beta = one @ qi @ Y / (one @ qi @ one)
    kk = k(np.array([1.3]), np.array([4.0]), OBS_X, OBS_T)
    assert p.values.ravel()[0] == pytest.approx(beta + (kk @ qi @ (Y - beta))[0], rel=1e-9)
    assert a.values.shape == (2, 1) and np.all(a.std > 0)
