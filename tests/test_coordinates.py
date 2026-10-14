import itertools

import numpy as np
import pytest
from scipy import integrate

import rebasis as rb


def test_gaussian_ils_in_nm_on_a_wavenumber_grid_matches_quadrature():
    nu = np.linspace(19_700, 20_300, 60001)
    hats = rb.Space.hats(nu)
    f = 1 + 0.5 * np.sin(nu / 3.0)
    ils = rb.Space.gaussians([499.4321, 500.1234], fwhm=0.5)
    g = rb.measure(ils.to_wavenumber(jacobian=True), hats) @ f

    # reference: the average over wavelength of the linear interpolant in wavenumber, by a dense
    # trapezoid rule (adaptive quadrature struggles with the interpolant's thousands of kinks)
    sigma = 0.5 / rb.FWHM_PER_SIGMA
    for gi, c in zip(g, [499.4321, 500.1234], strict=True):
        lam = np.linspace(c - 6 * sigma, c + 6 * sigma, 4_000_001)
        k = np.exp(-0.5 * ((lam - c) / sigma) ** 2)
        fl = np.interp(1e7 / lam, nu, f)
        assert gi == pytest.approx(np.trapezoid(k * fl, lam) / np.trapezoid(k, lam), rel=1e-10)


def test_jacobian_choice_matters_and_round_trips():
    lam = np.linspace(400, 410, 101)
    hats = rb.Space.hats(lam)
    a = hats.to_wavenumber(jacobian=True).integrals()
    b = hats.to_wavenumber(jacobian=False).integrals()
    np.testing.assert_allclose(a, hats.integrals(), rtol=1e-12)
    # without the jacobian, ∫ dν = ∫ |dν/dλ| dλ ≈ 1e7/λ² × ∫ dλ
    np.testing.assert_allclose(b / a, 1e7 / lam**2, rtol=1e-3)
    back = hats.to_wavenumber(jacobian=False).to_wavelength(jacobian=False)
    np.testing.assert_allclose(back.lower, hats.lower, rtol=1e-14)


def test_regrid_function_from_wavelength_to_wavenumber_boxes():
    lam = np.linspace(500, 510, 10001)
    f = np.cos(lam)  # a function of wavelength (e.g. a transmittance)
    src = rb.Space.hats(lam).to_wavenumber(jacobian=False)
    edges = np.linspace(1e7 / 509.0, 1e7 / 501.0, 41)
    g = rb.regrid(f, src, rb.Space.boxes(edges))
    # box averages over wavenumber of f(λ(ν))
    ref = [
        integrate.quad(lambda v: np.cos(1e7 / v), a, b, limit=200)[0] / (b - a)
        for a, b in itertools.pairwise(edges)
    ]
    np.testing.assert_allclose(g, ref, atol=2e-6)


def test_wavelength_derivatives_through_the_map():
    nu = np.linspace(19_800, 20_200, 8001)
    hats = rb.Space.hats(nu)
    f = np.exp(-(((nu - 20_000) / 30) ** 2))
    build = lambda c: rb.Space.gaussians([c], fwhm=0.4).to_wavenumber(jacobian=True)  # noqa: E731
    _, dw = rb.measure_derivative(build(499.87), hats, "center")
    h = 1e-5
    fd = (rb.measure(build(499.87 + h), hats) @ f - rb.measure(build(499.87 - h), hats) @ f) / (
        2 * h
    )
    np.testing.assert_allclose(dw @ f, fd, rtol=1e-5)


def test_jacobian_is_required():
    with pytest.raises(TypeError):
        rb.Space.hats([400.0, 401.0]).to_wavenumber()
