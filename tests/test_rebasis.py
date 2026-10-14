import numpy as np
import pytest
import scipy.sparse as sp

import rebasis as rb


def test_gaussian_ils_on_hats_matches_brute_force_convolution():
    x = np.linspace(500.0, 510.0, 20001)
    rng = np.random.default_rng(0)
    # a spectrum with narrow absorption lines
    lines = rng.uniform(501, 509, 40)
    f = 1.0 - sum(0.5 * np.exp(-0.5 * ((x - c) / 0.01) ** 2) for c in lines)
    centers = np.linspace(502, 508, 61)
    fwhm = 0.3
    w = rb.measure(rb.Space.gaussians(centers, fwhm=fwhm), rb.Space.hats(x))
    assert isinstance(w, sp.csr_array)
    assert w.shape == (61, 20001)
    g = w @ f

    # brute force: trapezoid integral of the linear interpolant times a unit-area Gaussian
    sigma = fwhm / rb.FWHM_PER_SIGMA
    xx = np.linspace(500, 510, 400001)
    ff = np.interp(xx, x, f)
    k = np.exp(-0.5 * ((xx[None, :] - centers[:, None]) / sigma) ** 2)
    k[np.abs(xx[None, :] - centers[:, None]) > 6 * sigma] = 0
    brute = np.trapezoid(k * ff, xx, axis=1) / np.trapezoid(k, xx, axis=1)
    np.testing.assert_allclose(g, brute, rtol=1e-8)


def test_matrix_is_sparse_and_banded():
    x = np.linspace(0, 100, 100001)
    w = rb.measure(rb.Space.gaussians(np.linspace(10, 90, 81), sigma=0.2), rb.Space.hats(x))
    # each 6-sigma Gaussian covers 2.4 units = ~2400 hats
    assert w.nnz < 81 * 2500


def test_projection_conserves_and_applies_along_any_axis():
    x = np.linspace(0, 1, 201)
    fine = rb.Space.hats(x)
    coarse = rb.Space.boxes(np.array([0, 0.1, 0.5, 0.55, 1.0]))
    p = rb.project(fine, coarse)
    f = np.stack([np.sin(5 * x), x**2, np.ones_like(x)])  # (3, 201)
    g = p.apply(f, axis=-1)
    assert g.shape == (3, 4)
    np.testing.assert_allclose(g @ coarse.integrals, f @ fine.integrals, rtol=1e-13)
    np.testing.assert_allclose(g[2], 1.0, rtol=1e-13)
    np.testing.assert_allclose(p.apply(f.T, axis=0), g.T, rtol=1e-14)
    np.testing.assert_allclose(p @ f.T, g.T, rtol=1e-14)
    np.testing.assert_allclose(f @ p.to_dense().T, g, rtol=1e-12, atol=1e-14)


def test_apply_along_matches_matmul():
    x = np.linspace(0, 1, 51)
    w = rb.measure(rb.Space.gaussians([0.3, 0.6], sigma=0.05), rb.Space.hats(x))
    f = np.random.default_rng(1).normal(size=(4, 51, 2))
    out = rb.apply_along(w, f, axis=1)
    assert out.shape == (4, 2, 2)
    np.testing.assert_allclose(out[2, :, 1], w @ f[2, :, 1])


def test_evaluate_is_linear_interpolation():
    nodes = np.array([0.0, 1.0, 3.0, 4.0])
    f = np.array([1.0, 2.0, 0.0, 5.0])
    xq = np.linspace(0, 4, 17)
    np.testing.assert_allclose(rb.Space.hats(nodes).evaluate(xq) @ f, np.interp(xq, nodes, f))


def test_shifted_tabulated_shape_with_stretch():
    shape_x = np.array([-1.0, 0.0, 1.0])
    shape_y = np.array([0.0, 1.0, 0.0])
    s = rb.Space.shifted([2.0, 5.0], shape_x, shape_y, stretch=[0.5, 2.0])
    np.testing.assert_allclose(s.integrals, [0.5, 2.0])
    np.testing.assert_allclose(s.lower, [1.5, 3.0])


def test_boxes_from_centers():
    s = rb.Space.boxes_from_centers([1.0, 2.0, 4.0])
    np.testing.assert_allclose(s.lower, [0.5, 1.5, 3.0])
    np.testing.assert_allclose(s.upper, [1.5, 3.0, 5.0])


def test_errors():
    with pytest.raises(ValueError, match="strictly increasing"):
        rb.Space.hats([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="exactly one"):
        rb.Space.gaussians([0.0])
    with pytest.raises(ValueError, match="normalization"):
        rb.measure(rb.Space.deltas([0.5]), rb.Space.hats([0, 1]), "bogus")
    dup = rb.Space.deltas([0.5, 0.5])
    with pytest.raises(rb.SingularError):
        rb.project(rb.Space.hats([0, 1]), dup)
