"""measure_derivative against finite differences of measure, through the Python API."""

import numpy as np
import pytest

import rebasis as rb

X = np.linspace(0, 2, 4001)
HATS = rb.Space.hats(X)
F = np.exp(-(((X - 1.0) / 0.3) ** 2)) * (1 + 0.3 * np.sin(40 * X))  # a structured spectrum
C = np.array([0.6123, 1.2377])

CASES = [
    ("center", 0.0, lambda t: rb.Space.voigts(C + t, sigma=0.03, gamma=0.01, cutoff=10)),
    (
        "scale",
        0.0,
        lambda t: rb.Space.voigts(C, sigma=0.03 * np.exp(t), gamma=0.01 * np.exp(t), cutoff=10),
    ),
    ("sigma", 0.03, lambda t: rb.Space.voigts(C, sigma=t, gamma=0.01, cutoff=10)),
    ("gamma", 0.01, lambda t: rb.Space.voigts(C, sigma=0.03, gamma=t, cutoff=10)),
    ("power", 3.0, lambda t: rb.Space.super_gaussians(C, fwhm=0.1, power=t)),
    ("asymmetry", 0.1, lambda t: rb.Space.super_gaussians(C, fwhm=0.1, power=3, asymmetry=t)),
    ("eta", 0.4, lambda t: rb.Space.pseudo_voigts(C, fwhm=0.06, eta=t, cutoff=10)),
    ("half_width", 0.05, lambda t: rb.Space.smoothed_boxes(C, width=2 * t, sigma=0.01)),
    (
        "center",
        0.0,
        lambda t: rb.Space.fts(C + t, max_opd=15.0, apodization="norton_beer_strong", cutoff=10),
    ),
    ("scale", 0.0, lambda t: rb.Space.shifted(C, [-0.05, 0, 0.08], [0, 1, 0], stretch=np.exp(t))),
]


@pytest.mark.parametrize(
    "parameter,t0,build", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)]
)
def test_jacobian_matches_finite_difference(parameter, t0, build):
    w, dw = rb.measure_derivative(build(t0), HATS, parameter)
    np.testing.assert_allclose((w @ F), rb.measure(build(t0), HATS) @ F, rtol=1e-14)
    h = 1e-6 * max(abs(t0), 1.0)
    fd = (rb.measure(build(t0 + h), HATS) @ F - rb.measure(build(t0 - h), HATS) @ F) / (2 * h)
    np.testing.assert_allclose(dw @ F, fd, rtol=1e-6, atol=1e-9 * np.abs(fd).max())


def test_unknown_parameter():
    with pytest.raises(ValueError, match="no parameter"):
        rb.measure_derivative(rb.Space.gaussians([1.0], sigma=0.1), HATS, "eta")
    with pytest.raises(ValueError, match="unknown parameter"):
        rb.measure_derivative(rb.Space.gaussians([1.0], sigma=0.1), HATS, "bogus")
