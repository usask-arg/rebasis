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


# ---------------------------------------------------------------- multiple columns, mean priors

TARGETS = {"x": rb.Space.boxes([0.0, 1.5, 3.0, 5.0]), "t": rb.Space.boxes([0.0, 4.0, 10.0])}


def _single(y, keep=None, noise=0.05**2, **kw):
    keep = np.ones(N, bool) if keep is None else keep
    nz = noise[keep] if np.ndim(noise) else noise
    return rb.optimal_estimate(
        y[keep], {"x": OBS_X[keep], "t": OBS_T[keep]}, TARGETS, COV, noise=nz, **kw
    )


def test_columns_sharing_observations_equal_separate_estimates():
    ys = np.column_stack([Y, 2 * Y - 1, np.sin(OBS_X)])
    est = rb.optimal_estimate(
        ys, {"x": OBS_X, "t": OBS_T}, TARGETS, COV, noise=0.05**2, full_covariance=True
    )
    assert est.values.shape == (3, 2, 3) and est.weights.shape == (6, N)
    for j in range(3):
        one = _single(ys[:, j], full_covariance=True)
        np.testing.assert_allclose(est.values[..., j], one.values, rtol=1e-12)
        np.testing.assert_allclose(est.std[..., j], one.std, rtol=1e-12)
        assert est.log_likelihood[j] == pytest.approx(one.log_likelihood, rel=1e-12)
        assert est.mean[j] == pytest.approx(one.mean, rel=1e-12)
    np.testing.assert_allclose(est.apply(ys), est.values, rtol=1e-12)
    np.testing.assert_allclose(est.covariance, one.covariance, rtol=1e-12)


def test_columns_with_different_missing_data_and_noise():
    rng = np.random.default_rng(3)
    ys = np.column_stack([Y, Y + 0.3, Y - 0.2])
    ys[rng.choice(N, 10, replace=False), 1] = np.nan
    noise = np.full((N, 3), 0.05**2)
    noise[:, 2] = 0.1**2
    est = rb.optimal_estimate(ys, {"x": OBS_X, "t": OBS_T}, TARGETS, COV, noise=noise)
    assert est.weights.shape == (3, 6, N)
    for j in range(3):
        keep = np.isfinite(ys[:, j])
        one = _single(ys[:, j], keep, noise=noise[:, j])
        np.testing.assert_allclose(est.values[..., j], one.values, rtol=1e-12)
        np.testing.assert_allclose(est.std[..., j], one.std, rtol=1e-12)
        assert np.all(est.weights[j][:, ~keep] == 0)
    np.testing.assert_allclose(est.apply(ys), est.values, rtol=1e-12)
    ll = rb.log_likelihood(ys, {"x": OBS_X, "t": OBS_T}, COV, noise=noise)
    np.testing.assert_allclose(ll, est.log_likelihood, rtol=1e-12)


def test_finite_prior_mean_term_matches_augmented_covariance():
    # constant (unconstrained) + a linear drift in t with prior std 0.4 on each hat coefficient
    hats = rb.Space.hats([0.0, 10.0])
    mean = [rb.MeanTerm({}), rb.MeanTerm({"t": hats}, prior_std=0.4)]
    gx, gt = np.array([1.0, 3.5]), np.array([2.0, 8.0])
    est = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.deltas(gx), "t": rb.Space.deltas(gt)},
        COV,
        noise=0.05**2,
        mean=mean,
    )
    hp = np.column_stack([1 - OBS_T / 10, OBS_T / 10])
    xx, tt = np.meshgrid(gx, gt, indexing="ij")
    hp_t = np.column_stack([1 - tt.ravel() / 10, tt.ravel() / 10])
    q = kernel(OBS_X, OBS_T, OBS_X, OBS_T) + 0.4**2 * hp @ hp.T + 0.05**2 * np.eye(N)
    k = kernel(xx.ravel(), tt.ravel(), OBS_X, OBS_T) + 0.4**2 * hp_t @ hp.T
    qi = np.linalg.inv(q)
    one = np.ones(N)
    beta = one @ qi @ Y / (one @ qi @ one)
    g = beta + k @ qi @ (Y - beta)
    np.testing.assert_allclose(est.values.ravel(), g, rtol=1e-9)
    prior = 2.0 + 0.4**2 * np.einsum("mp,mp->m", hp_t, hp_t)
    r = 1 - k @ qi @ one
    var = prior - np.einsum("ij,jk,ik->i", k, qi, k) + r**2 / (one @ qi @ one)
    np.testing.assert_allclose(est.std.ravel(), np.sqrt(var), rtol=1e-7)


def test_prior_limits():
    hats = rb.Space.hats([0.0, 10.0])
    flat = _single(Y, mean={"t": hats})
    # the flat limit is approached as (field variance / prior variance); far larger priors only
    # cost precision in the factorization, which is what prior_std=inf avoids
    broad = _single(Y, mean=[rb.MeanTerm({"t": hats}, prior_std=1e3)])
    np.testing.assert_allclose(broad.values, flat.values, rtol=1e-5)
    zero = _single(Y, mean=[rb.MeanTerm({}), rb.MeanTerm({"t": hats}, prior_std=0.0)])
    np.testing.assert_allclose(zero.values, _single(Y).values, rtol=1e-12)


def test_prior_makes_bases_without_data_usable():
    # spline functions beyond x = 5 have no data: singular unconstrained, fine with a prior
    xb = rb.Space.bsplines_clamped(np.linspace(0, 8, 9), 3)
    with pytest.raises(rb.SingularError, match="prior_std"):
        _single(Y, mean={"x": xb})
    est = _single(Y, mean=[rb.MeanTerm({}), rb.MeanTerm({"x": xb}, prior_std=1.0)])
    assert np.all(np.isfinite(est.values)) and np.all(est.std > 0)


# ------------------------------------------------------------------------- neighbourhoods


def test_neighborhood_covering_everything_is_the_global_estimate():
    glob = _single(Y)
    loc = _single(Y, neighborhood={"x": 100.0, "t": 100.0})
    np.testing.assert_allclose(loc.values, glob.values, rtol=1e-12)
    np.testing.assert_allclose(loc.std, glob.std, rtol=1e-12)
    np.testing.assert_allclose(loc.weights, glob.weights, atol=1e-14)
    assert np.all(loc.n_used == N) and np.isnan(loc.log_likelihood)


def test_each_block_is_the_estimate_from_its_own_observations():
    half = 0.8
    edges = np.array([0.0, 1.5, 3.0, 5.0])
    loc = _single(Y, neighborhood={"x": half})
    for i, (a, b) in enumerate(itertools.pairwise(edges)):
        keep = (OBS_X >= a - half) & (OBS_X <= b + half)
        ref = rb.optimal_estimate(
            Y[keep],
            {"x": OBS_X[keep], "t": OBS_T[keep]},
            {"x": rb.Space.boxes([a, b]), "t": TARGETS["t"]},
            COV,
            noise=0.05**2,
        )
        np.testing.assert_allclose(loc.values[i], ref.values[0], rtol=1e-12)
        np.testing.assert_allclose(loc.std[i], ref.std[0], rtol=1e-12)
        assert np.all(loc.n_used[i] == keep.sum())
        rows = np.ravel_multi_index((np.full(2, i), np.arange(2)), (3, 2))
        np.testing.assert_allclose(loc.weights[rows][:, keep], ref.weights, atol=1e-14)
        assert np.all(loc.weights[rows][:, ~keep] == 0)
    np.testing.assert_allclose(loc.apply(Y), loc.values, rtol=1e-12)


def test_periodic_neighborhood_wraps():
    lon = np.array([2.0, 5.0, 120.0, 200.0, 355.0])
    cov = rb.Covariance(1.0, lon=rb.CovarianceDim("exponential", 30.0, period=360.0))
    est = rb.optimal_estimate(
        np.arange(5.0),
        {"lon": lon},
        {"lon": rb.Space.boxes([340.0, 360.0, 380.0])},
        cov,
        noise=0.01,
        neighborhood={"lon": 10.0},
    )
    # [330, 370] reaches 2, 5 (as 362, 365) and 355; [350, 390] the same three
    np.testing.assert_array_equal(est.n_used, [3, 3])


def test_empty_blocks_are_nan():
    est = _single(Y, neighborhood={"t": 0.0}, mean="estimate")
    assert np.all(est.n_used > 0)
    far = rb.optimal_estimate(
        Y,
        {"x": OBS_X, "t": OBS_T},
        {"x": rb.Space.boxes([0.0, 5.0]), "t": rb.Space.boxes([0.0, 10.0, 50.0, 60.0])},
        COV,
        noise=0.05**2,
        neighborhood={"t": 1.0},
    )
    assert np.isfinite(far.values[0, 0]) and np.isnan(far.values[0, 2])
    assert far.n_used[0, 2] == 0


def test_global_basis_is_trimmed_per_block():
    # splines over [0, 20]: globally the functions beyond the data are undetermined
    xb = rb.Space.bsplines_clamped(np.linspace(0, 20, 21), 3)
    with pytest.raises(rb.SingularError):
        _single(Y, mean={"x": xb})
    est = _single(Y, mean={"x": xb}, neighborhood={"x": 0.5})
    assert np.all(np.isfinite(est.values))
    assert est.mean_coefficients is None


def test_neighborhood_errors():
    with pytest.raises(ValueError, match="full_covariance"):
        _single(Y, neighborhood={"x": 1.0}, full_covariance=True)
    with pytest.raises(ValueError, match="unknown"):
        _single(Y, neighborhood={"z": 1.0})


# ------------------------------------------------------------------- periodic mean bases


def test_zonal_mean_from_one_longitude_sector_with_a_wave_mean():
    # a latitude structure plus waves 1 and 2 with latitude-dependent amplitudes, sampled only in
    # 200-300°E: with the waves in the mean (and the field in its span), the area-weighted zonal
    # means are recovered exactly from that one sector
    rng = np.random.default_rng(4)
    n = 300
    lat = rng.uniform(-60, 60, n)
    lon = rng.uniform(200, 300, n)
    lat_b = rb.Space.bsplines_clamped(np.linspace(-60, 60, 5), 3)
    waves = rb.Space.harmonics(2, 360.0)
    c0 = rng.uniform(1, 3, len(lat_b))  # the zonal-mean structure
    c1 = rng.normal(0, 0.5, (len(lat_b), len(waves)))  # wave amplitudes by latitude
    bl = lat_b.evaluate(lat).toarray()
    bw = waves.evaluate(lon).toarray()
    y = bl @ c0 + np.einsum("ni,ij,nj->n", bl, c1, bw)
    cov = rb.Covariance(
        0.01,
        lat=rb.CovarianceDim("squared_exponential", 10.0),
        lon=rb.CovarianceDim("exponential", 30.0, period=360.0),
    )
    edges = [-40.0, -20.0, 20.0]
    est = rb.optimal_estimate(
        y,
        {"lat": lat, "lon": lon},
        {"lat": rb.Space.boxes(edges), "lon": rb.Space.boxes([0, 360])},
        cov,
        noise=1e-8,
        target_weight={"lat": "cos_lat_deg"},
        mean=[rb.MeanTerm({"lat": lat_b}), rb.MeanTerm({"lat": lat_b, "lon": waves})],
    )
    truth = rb.measure(rb.Space.boxes(edges), lat_b, weight="cos_lat_deg") @ c0
    np.testing.assert_allclose(est.values[:, 0], truth, atol=1e-9)
    naive = [y[(lat >= a) & (lat < b)].mean() for a, b in [(-40, -20), (-20, 20)]]
    assert np.max(np.abs(np.array(naive) - truth)) > 0.05  # the sector alone is biased


def test_periodic_bsplines_in_python():
    s = rb.Space.bsplines_periodic(np.linspace(0, 360, 9), 3)
    assert len(s) == 8
    x = np.linspace(0, 359.9, 500)
    np.testing.assert_allclose(s.evaluate(x).toarray().sum(axis=1), 1.0, atol=1e-13)
    h = rb.Space.harmonics(3)
    assert len(h) == 6
    np.testing.assert_allclose(
        h.evaluate([90.0]).toarray().ravel(), [0, 1, -1, 0, 0, -1], atol=1e-14
    )
