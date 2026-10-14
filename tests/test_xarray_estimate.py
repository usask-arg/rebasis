"""rebasis.xarray.optimal_estimate / fit_covariance: the same numbers as the array API, with
labelled dimensions, datetimes and extra (column) dimensions handled."""

import numpy as np
import pytest

import rebasis as rb

xr = pytest.importorskip("xarray")
import rebasis.xarray as rbx  # noqa: E402

RNG = np.random.default_rng(3)
N = 150
START = np.datetime64("2010-09-01")
TIME = START + (RNG.uniform(0, 30, N) * 86400e3).astype("timedelta64[ms]")
LAT = RNG.uniform(-60, 60, N)
LON = RNG.uniform(-180, 180, N)
ALT = np.array([20.0, 30.0, 40.0])
VALUES = (
    np.cos(np.radians(LAT))[:, None] * ALT[None, :] / 10
    + 0.01 * (TIME - START).astype(float)[:, None] / 86400e3
    + RNG.normal(0, 0.05, (N, 3))
)
VALUES[5, 1] = np.nan  # an invalid observation in one column
ERROR = np.full((N, 3), 0.05)

COV = rb.Covariance(
    1.0,
    lat=rb.CovarianceDim("squared_exponential", 20.0),
    lon=rb.CovarianceDim("exponential", 60.0, period=360.0),
    time=rb.CovarianceDim("matern32", 10.0),
)
EDGES = np.array([-60.0, -20.0, 20.0, 60.0])
MONTH = np.array(["2010-09-01", "2010-10-01"], dtype="datetime64[ns]")


def dataset():
    return xr.Dataset(
        {
            "ozone": (("time", "altitude"), VALUES, {"units": "ppmv"}),
            "ozone_error": (("time", "altitude"), ERROR),
            "latitude": ("time", LAT),
            "longitude": ("time", LON % 360),
        },
        coords={"time": TIME, "altitude": ALT},
    )


def numpy_inputs():
    day = (TIME - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    coords = {"lat": LAT, "lon": LON % 360, "time": day}
    month = (MONTH - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    targets = {
        "lat": rb.Space.boxes(EDGES),
        "lon": rb.Space.boxes([0.0, 360.0]),
        "time": rb.Space.boxes(month),
    }
    return coords, targets


def test_dataset_matches_the_array_api():
    ds = dataset()
    out, est = rbx.optimal_estimate(
        ds,
        {"lat": EDGES, "lon": [0.0, 360.0], "time": MONTH},
        COV,
        var="ozone",
        noise=ds.ozone_error**2,
        coords={"lat": "latitude", "lon": "longitude"},
        target_weight={"lat": "cos_lat_deg"},
        return_estimate=True,
    )
    coords, targets = numpy_inputs()
    ref = rb.optimal_estimate(
        VALUES, coords, targets, COV, noise=ERROR**2, target_weight={"lat": "cos_lat_deg"}
    )
    assert out.ozone.dims == ("lat", "lon", "time", "altitude")
    np.testing.assert_allclose(out.ozone.values, ref.values, rtol=1e-12)
    np.testing.assert_allclose(out.ozone_std.values, ref.std, rtol=1e-12)
    np.testing.assert_allclose(out.prior_std.values, ref.prior_std, rtol=1e-12)
    np.testing.assert_allclose(out.log_likelihood.values, ref.log_likelihood, rtol=1e-12)
    np.testing.assert_allclose(out.constant_mean.values, ref.mean, rtol=1e-12)
    np.testing.assert_array_equal(out.n_used.sel(altitude=30.0).values, N - 1)
    np.testing.assert_array_equal(out.n_used.sel(altitude=20.0).values, N)
    assert est.weights.shape == (3, 3, N)  # columns differ in their valid observations
    # labelled coordinates
    np.testing.assert_array_equal(out.lat.values, [-40.0, 0.0, 40.0])
    np.testing.assert_array_equal(out.lat_bounds.values[0], [-60.0, -20.0])
    assert out.time.values[0] == np.datetime64("2010-09-16")
    np.testing.assert_array_equal(out.time_bounds.values[0], MONTH)
    np.testing.assert_array_equal(out.altitude.values, ALT)
    assert out.ozone.attrs["units"] == "ppmv"
    assert out.lat.attrs["bounds"] == "lat_bounds"
    assert out.time.encoding["units"] == "days since 1970-01-01T00:00:00"


def test_noise_by_name_and_a_fit():
    ds = dataset().isel(altitude=0).rename(latitude="lat", longitude="lon")
    ds["noise"] = ds.ozone_error**2
    fit = rbx.fit_covariance(ds, COV, var="ozone", noise="noise", maxiter=30)
    coords, targets = numpy_inputs()
    ref_fit = rb.fit_covariance(VALUES[:, 0], coords, COV, noise=ERROR[:, 0] ** 2, maxiter=30)
    assert fit.parameters == pytest.approx(ref_fit.parameters, rel=1e-10)
    # a CovarianceFit brings its mean and nugget
    out = rbx.optimal_estimate(
        ds, {"lat": EDGES, "lon": [0.0, 360.0], "time": MONTH}, fit, var="ozone", noise="noise"
    )
    ref = fit.estimate(VALUES[:, 0], coords, targets, noise=ERROR[:, 0] ** 2)
    np.testing.assert_allclose(out.ozone.values, ref.values, rtol=1e-12)
    assert out.ozone.dims == ("lat", "lon", "time")
    assert out.log_likelihood.dims == ()


def test_dataarray_with_spaces_and_float_time():
    da = xr.DataArray(
        VALUES[:, 0],
        dims="obs",
        coords={"x": ("obs", LAT), "t": ("obs", np.linspace(0, 30, N))},
        name=None,
    )
    cov = rb.Covariance(
        1.0, x=rb.CovarianceDim("matern52", 15.0), t=rb.CovarianceDim("matern32", 10.0)
    )
    targets = {"x": rb.Space.hats(np.linspace(-60, 60, 7)), "t": [0.0, 15.0, 30.0]}
    out = rbx.optimal_estimate(da, targets, cov, noise=0.01, mean={"x": rb.Space.hats([-60, 60])})
    ref = rb.optimal_estimate(
        VALUES[:, 0],
        {"x": LAT, "t": np.linspace(0, 30, N)},
        {"x": targets["x"], "t": rb.Space.boxes([0.0, 15.0, 30.0])},
        cov,
        noise=0.01,
        mean={"x": rb.Space.hats([-60, 60])},
    )
    np.testing.assert_allclose(out.estimate.values, ref.values, rtol=1e-12)
    np.testing.assert_allclose(out.x_bounds.values[1], [-60.0, -20.0])
    assert "constant_mean" not in out  # not a single constant


def test_numeric_time_round_trip():
    t = rbx.numeric_time(MONTH, epoch="2010-01-01")
    np.testing.assert_allclose(t, [243.0, 273.0])
    np.testing.assert_allclose(rbx.numeric_time(MONTH, epoch="2010-09-01", unit="h"), [0, 720])
    np.testing.assert_array_equal(rbx._from_numeric(t, "2010-01-01", "D"), MONTH)


def test_errors():
    ds = dataset()
    targets = {"lat": EDGES, "lon": [0.0, 360.0], "time": MONTH}
    names = {"lat": "latitude", "lon": "longitude"}
    with pytest.raises(ValueError, match="pass var"):
        rbx.optimal_estimate(ds, targets, COV, noise=1.0, coords=names)
    with pytest.raises(ValueError, match="no variable or coordinate 'lat'"):
        rbx.optimal_estimate(ds, targets, COV, var="ozone", noise=1.0)
    with pytest.raises(TypeError, match="must be too"):
        rbx.optimal_estimate(
            ds, {**targets, "time": [0.0, 30.0]}, COV, var="ozone", noise=1.0, coords=names
        )
    with pytest.raises(ValueError, match="one column"):
        rbx.fit_covariance(ds, COV, var="ozone", noise=1.0, coords=names)
    with pytest.raises(ValueError, match="not covariance dimensions"):
        rbx.optimal_estimate(ds, targets, COV, var="ozone", noise=1.0, coords={"z": "x"})
