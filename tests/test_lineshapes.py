"""Line shapes against independent references (scipy special functions and quadrature)."""

import numpy as np
import pytest
from scipy import integrate, interpolate, special

import rebasis as rb

X = np.linspace(-3.0, 3.0, 1201)


def values(space: rb.Space, x=X) -> np.ndarray:
    """The (peak-1) shape of the single element of ``space`` at x."""
    return space.evaluate(x).toarray()[:, 0]


def test_faddeeva_matches_scipy_wofz():
    rng = np.random.default_rng(0)
    for x, y in zip(rng.normal(0, 5, 200), 10 ** rng.uniform(-6, 2, 200), strict=True):
        ref = special.wofz(x + 1j * y).real
        assert abs(rb.faddeeva_re(x, y) - ref) < 5e-15 * special.wofz(1j * y).real


@pytest.mark.parametrize("sigma,gamma", [(0.3, 1e-4), (0.3, 0.1), (0.05, 0.4), (0.2, 2.0)])
def test_voigt_shape_matches_scipy(sigma, gamma):
    s = rb.Space.voigts([0.0], sigma=sigma, gamma=gamma, cutoff=1e4)
    ref = special.voigt_profile(X, sigma, gamma) / special.voigt_profile(0.0, sigma, gamma)
    np.testing.assert_allclose(values(s), ref, rtol=1e-12, atol=1e-15)


def test_closed_form_shapes():
    np.testing.assert_allclose(
        values(rb.Space.lorentzians([0.0], fwhm=0.4, cutoff=1e3)), 1 / (1 + (X / 0.2) ** 2)
    )
    np.testing.assert_allclose(
        values(rb.Space.gaussians([0.0], fwhm=0.5)),
        np.where(np.abs(X) <= 6 * 0.5 / rb.FWHM_PER_SIGMA, np.exp(-4 * np.log(2) * X**2 / 0.25), 0),
        rtol=1e-14,
        atol=1e-300,
    )
    pv = rb.Space.pseudo_voigts([0.0], hwhm=0.3, eta=0.25)
    ref = 0.25 / (1 + (X / 0.3) ** 2) + 0.75 * np.exp(-np.log(2) * (X / 0.3) ** 2)
    np.testing.assert_allclose(values(pv), ref, rtol=1e-14)
    sg = rb.Space.super_gaussians([0.0], fwhm=1.0, power=4, asymmetry=0.2)
    w = np.where(X < 0, 0.4, 0.6)
    np.testing.assert_allclose(values(sg), np.exp(-np.log(2) * np.abs(X / w) ** 4), atol=1e-15)
    sb = rb.Space.smoothed_boxes([0.0], width=1.0, sigma=0.1)
    s2 = 0.1 * np.sqrt(2)
    ref = special.erfc((X - 0.5) / s2) - special.erfc((X + 0.5) / s2)
    ref = np.where(np.abs(X) <= 0.5 + 6 * 0.1, ref, 0.0)  # truncated 6σ beyond the edges
    np.testing.assert_allclose(values(sb), ref / ref.max(), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    "name,window",
    [
        ("none", lambda u: np.ones_like(u)),
        ("triangle", lambda u: 1 - np.abs(u)),
        ("hann", lambda u: 0.5 + 0.5 * np.cos(np.pi * u)),
        ("hamming", lambda u: 0.54 + 0.46 * np.cos(np.pi * u)),
        ("blackman", lambda u: 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2 * np.pi * u)),
        (
            "norton_beer_medium",
            lambda u: 0.152442 - 0.136176 * (1 - u**2) + 0.983734 * (1 - u**2) ** 2,
        ),
    ],
)
def test_fts_shapes_are_the_fourier_transform_of_the_window(name, window):
    opd = 2.0
    s = rb.Space.fts([0.0], max_opd=opd, apodization=name, cutoff=1e3)
    nu = np.linspace(-2, 2, 41)
    ref = np.array(
        [
            integrate.quad(
                lambda u, v=v: window(u) * np.cos(2 * np.pi * opd * v * u),
                -1,
                1,
                epsabs=1e-13,
                epsrel=1e-12,
                limit=200,
            )[0]
            for v in nu
        ]
    )
    np.testing.assert_allclose(values(s, nu), ref / ref[20], atol=1e-12)


def test_custom_apodization_matches_named():
    nu = np.linspace(-1, 1, 21)
    a = values(rb.Space.fts([0.0], max_opd=3.0, apodization="hamming"), nu)
    b = values(rb.Space.fts([0.0], max_opd=3.0, apodization={"cosine_sum": [0.54, 0.46]}), nu)
    np.testing.assert_allclose(a, b, rtol=1e-15)


ALL_KERNELS = [
    rb.Space.gaussians([0.3], sigma=0.2),
    rb.Space.lorentzians([0.3], hwhm=0.1, cutoff=50),
    rb.Space.voigts([0.3], sigma=0.1, gamma=0.05, cutoff=50),
    rb.Space.pseudo_voigts([0.3], hwhm=0.1, eta=0.5, cutoff=50),
    rb.Space.super_gaussians([0.3], hwhm=0.2, power=2.7, asymmetry=-0.3),
    rb.Space.smoothed_boxes([0.3], width=0.3, fwhm=0.1),
    rb.Space.fts([0.3], max_opd=5.0, apodization="norton_beer_strong", cutoff=30),
    rb.Space.fts([0.3], max_opd=5.0, apodization="blackman_harris", cutoff=30),
    rb.Space.rectangles([0.3], width=0.4),
    rb.Space.triangles([0.3], fwhm=0.4),
    rb.Space.trapezoids([0.3], top_width=0.2, bottom_width=0.6),
    rb.Space.shifted([0.3], [-0.5, -0.1, 0.0, 0.3, 0.5], [0, 0.7, 1, 0.4, 0], interp="pchip"),
]


@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=lambda s: s.kind)
def test_inner_products_with_a_box_match_quadrature(kernel):
    # ∫ K(x) dx over [a, b] = ⟨K, box⟩, against adaptive quadrature of the point values
    a, b = 0.17, 0.52
    got = rb.inner_products(kernel, rb.Space.boxes([a, b])).toarray()[0, 0]
    lo, hi = max(a, kernel.lower[0]), min(b, kernel.upper[0])
    f = lambda x: kernel.evaluate([x]).toarray()[0, 0]  # noqa: E731
    pts = [p for p in [0.3, 0.1, 0.2, 0.15, 0.45, 0.4, 0.5, 0.6, 0.8] if lo < p < hi]
    ref = integrate.quad(f, lo, hi, points=pts, epsabs=1e-15, epsrel=1e-13, limit=500)[0]
    assert abs(got - ref) < 1e-12 * abs(ref)
    # total area too
    lo, hi = kernel.lower[0], kernel.upper[0]
    pts = [p for p in [0.3, 0.1, 0.2, 0.15, 0.45, 0.4, 0.5, 0.6, 0.8] if lo < p < hi]
    ref = integrate.quad(f, lo, hi, points=pts, epsabs=1e-15, epsrel=1e-13, limit=2000)[0]
    assert abs(kernel.integrals()[0] - ref) < 1e-11 * abs(ref)


def test_bsplines_match_scipy():
    knots = np.array([0, 0, 0, 0.5, 1.2, 2.0, 2.0, 3.0, 3.5, 3.5, 3.5])
    x = np.linspace(0, 3.499, 300)
    for k in (1, 2, 3):
        t = (
            knots
            if k == 2
            else np.concatenate([[0] * (k - 2), knots, [3.5] * (k - 2)])
            if k > 2
            else knots[1:-1]
        )
        ref = interpolate.BSpline.design_matrix(x, t, k).toarray()
        got = rb.Space.bsplines(t, k).evaluate(x).toarray()
        np.testing.assert_allclose(got, ref, atol=1e-14)


def test_pchip_shape_matches_scipy():
    sx = np.array([-1.0, -0.4, 0.0, 0.2, 0.9])
    sy = np.array([0.0, 0.6, 1.0, 0.8, 0.0])
    s = rb.Space.shifted([0.0], sx, sy, interp="pchip")
    x = np.linspace(-0.999, 0.899, 200)
    np.testing.assert_allclose(values(s, x), interpolate.PchipInterpolator(sx, sy)(x), atol=1e-15)


def test_per_channel_tabulated_shapes():
    x = np.linspace(-1, 1, 21)
    y = np.stack([np.exp(-((x / w) ** 2)) for w in (0.2, 0.4)])
    s = rb.Space.tabulated([5.0, 7.0], x, y)
    np.testing.assert_allclose(s.lower, [4.0, 6.0])
    widths = s.integrals()
    assert widths[1] > 1.9 * widths[0]


def test_every_kernel_returns_the_center_value_of_a_linear_spectrum():
    x = np.linspace(-10, 10, 200001)  # covers every kernel's support
    f = 2 * x + 1
    hires = rb.Space.hats(x)
    for k in ALL_KERNELS:
        if k.kind in ("super_gaussians", "shifted"):  # asymmetric: centroid is not the center
            continue
        g = rb.measure(k, hires) @ f
        assert abs(g[0] - (2 * 0.3 + 1)) < 1e-11, k.kind
