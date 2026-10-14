import numpy as np
import pytest

import rebasis as rb


def test_latlon_area_weighted_regrid_conserves_global_mean():
    lat_f = rb.Space.boxes(np.linspace(-90, 90, 181))
    lon_f = rb.Space.boxes(np.linspace(0, 360, 361))
    lat_c = rb.Space.boxes(np.linspace(-90, 90, 13))
    lon_c = rb.Space.boxes(np.linspace(0, 360, 25))
    lat, lon = np.meshgrid(lat_f.centers, lon_f.centers, indexing="ij")
    f = np.cos(np.radians(lat)) ** 2 * (1 + 0.3 * np.sin(np.radians(3 * lon))) + 0.1 * lat / 90
    g = rb.regrid_nd(f, {0: (lat_f, lat_c, "cos_lat_deg"), 1: (lon_f, lon_c)})
    assert g.shape == (12, 24)

    def global_mean(v, lat_s, lon_s):
        w = np.outer(lat_s.integrals("cos_lat_deg"), lon_s.integrals())
        return (v * w).sum() / w.sum()

    assert abs(global_mean(f, lat_f, lon_f) - global_mean(g, lat_c, lon_c)) < 1e-14


def test_skipna_box_average_ignores_missing():
    src = rb.Space.boxes(np.arange(0.0, 9.0))
    dst = rb.Space.boxes([0.0, 4.0, 8.0])
    f = np.array([1.0, np.nan, 3.0, 5.0, np.nan, np.nan, np.nan, 7.0])
    g = rb.regrid(f, src, dst, skipna=True, na_thres=0.2)
    np.testing.assert_allclose(g, [3.0, 7.0])
    g = rb.regrid(f, src, dst, skipna=True, na_thres=0.5)
    assert g[0] == pytest.approx(3.0) and np.isnan(g[1])


def test_bounded_projection_has_no_overshoot():
    x = np.linspace(0, 1, 2001)
    f = (np.abs(x - 0.5) < 0.13).astype(float)
    src = rb.Space.hats(x)
    dst = rb.Space.bsplines_clamped(np.linspace(0, 1, 15), 3)
    plain = rb.regrid(f, src, dst)
    bounded = rb.regrid(f, src, dst, bounded=True)
    assert plain.min() < -1e-3 and plain.max() > 1 + 1e-3
    assert bounded.min() >= -1e-15 and bounded.max() <= 1 + 1e-15
    m = dst.integrals()
    np.testing.assert_allclose(plain @ m, bounded @ m, rtol=1e-13)
    np.testing.assert_allclose(bounded @ m, f @ src.integrals(), rtol=1e-12)


def test_fit_upsamples_bin_averages_conservatively():
    edges = np.linspace(0, 2 * np.pi, 13)
    bins = rb.Space.boxes(edges)
    avg = np.diff(-np.cos(edges)) / np.diff(edges)  # exact bin averages of sin
    model = rb.Space.bsplines_clamped(edges, 3)
    c = rb.regrid(avg, bins, model, method="fit", smoothing=1e-8)
    fitted_avg = rb.measure(bins, model) @ c
    np.testing.assert_allclose(fitted_avg, avg, atol=1e-6)
    xq = np.linspace(0.5, 5.8, 50)
    np.testing.assert_allclose(model.evaluate(xq) @ c, np.sin(xq), atol=5e-3)


def test_fit_to_samples_is_scipy_least_squares_spline_and_smoothing_helps():
    from scipy import interpolate

    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 1, 400))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, x.size)
    bp = np.linspace(0, 1, 30)
    model = rb.Space.bsplines_clamped(bp, 3)
    c = rb.fit(rb.Space.deltas(x), model, y, normalization="none")
    ref = interpolate.make_lsq_spline(x, y, np.concatenate([[0] * 3, bp, [1] * 3]), k=3)
    np.testing.assert_allclose(c, ref.c, atol=1e-13)

    xq = np.linspace(0, 1, 101)
    err = {
        s: np.abs(
            model.evaluate(xq)
            @ rb.fit(rb.Space.deltas(x), model, y, normalization="none", smoothing=s)
            - np.sin(2 * np.pi * xq)
        ).max()
        for s in (0.0, 1e-2)
    }
    assert err[1e-2] < 0.6 * err[0.0]


def test_interpolate_method():
    src = rb.Space.hats([0.0, 1.0, 2.0])
    dst = rb.Space.deltas([0.5, 1.5])
    np.testing.assert_allclose(rb.regrid([0.0, 2.0, 0.0], src, dst, method="interpolate"), [1, 1])


def test_xarray_dataset_regrid():
    xr = pytest.importorskip("xarray")
    import rebasis.xarray  # noqa: F401

    lat = np.linspace(89.5, -89.5, 180)  # decreasing, as in many products
    time = np.arange(3)
    da = xr.DataArray(
        np.cos(np.radians(lat))[None, :] * (1 + time[:, None]),
        dims=("time", "lat"),
        coords={"time": time, "lat": lat, "lat_bnds_center": ("lat", lat)},
        attrs={"units": "K"},
        name="t",
    )
    ds = xr.Dataset({"t": da, "other": ("time", time * 1.0)})
    out = rb.xarray.regrid(ds, {"lat": np.arange(-85.0, 90.0, 10.0)}, weight={"lat": "cos_lat_deg"})
    assert out["t"].dims == ("time", "lat") and out.sizes["lat"] == 18
    assert out["t"].attrs == {"units": "K"}
    np.testing.assert_array_equal(out["other"].values, time * 1.0)
    assert "lat_bnds_center" not in out.coords
    # the area-weighted mean is unchanged
    src = rb.Space.boxes_from_centers(lat[::-1])
    w_src = src.integrals("cos_lat_deg")
    w_dst = rb.Space.boxes_from_centers(np.arange(-85.0, 90.0, 10.0)).integrals("cos_lat_deg")
    before = (da.sortby("lat").values * w_src).sum(-1) / w_src.sum()
    after = (out["t"].values * w_dst).sum(-1) / w_dst.sum()
    np.testing.assert_allclose(after, before, rtol=1e-13)
