"""Monthly zonal means of real OSIRIS ozone at one altitude: bin averages vs rebasis optimal
estimates (with uncertainties). There is no truth here; see sampling_study.py for that.

    uv run --with h5py --with h5netcdf python examples/osiris/osiris_month.py 2010 9 20
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import sampling_study as ss
from data import days_in_month, load_osiris, month_start

import rebasis as rb

AVOGADRO = 6.02214076e23


def start_covariance(v: float, lat_range) -> rb.Covariance:
    """Correlation lengths in latitude (°), longitude (°) and time (days), with the standard
    deviation a smooth function of latitude: the starting point of the fit."""
    lo, hi = lat_range
    basis = rb.Space.bsplines_clamped(np.linspace(lo, hi, round((hi - lo) / 30) + 1), 3)
    profile = rb.VarianceProfile(basis, np.full(len(basis), np.sqrt(v)))
    return rb.Covariance(
        1.0,
        lat=rb.CovarianceDim("squared_exponential", 10.0, variance_profile=profile),
        lon=rb.CovarianceDim("exponential", 60.0, period=360.0),
        time=rb.CovarianceDim("matern32", 5.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("month", type=int)
    ap.add_argument("altitude", type=float, help="km")
    ap.add_argument("--knots", type=float, default=15.0)
    ap.add_argument("--out", default=str(Path(__file__).with_name("out")))
    args = ap.parse_args()
    ss.LAT_KNOT_SPACING = args.knots
    ss.DRIFT_PRIOR = 0.1  # a starting value: the drift's prior width is fitted
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ndays = days_in_month(args.year, args.month)
    t0 = month_start(args.year, args.month)

    ds = load_osiris(args.year, args.month).sel(altitude=args.altitude, method="nearest")
    # number density in 10^12 cm^-3, longitudes in [0, 360)
    scale = AVOGADRO * 1e-6 / 1e12
    ds = ds.assign(
        o3=ds.ozone_concentration * scale,
        o3_var=(ds.ozone_concentration_standard_error * scale) ** 2,
        longitude=ds.longitude % 360,
    )
    ds = ds.where(np.isfinite(ds.o3) & (ds.o3_var > 0), drop=True)
    lat = ds.latitude.values
    print(f"{ds.sizes['time']} OSIRIS profiles at {float(ds.altitude):.0f} km")

    # Times become days since the start of the month (epoch=t0), the units of the covariance's
    # time length and of the mean basis.
    names = {"lat": "latitude", "lon": "longitude"}
    lat_range = ss.coverage(lat)
    tic = time.perf_counter()
    fit = rb.xarray.fit_covariance(
        ds,
        start_covariance(float(ds.o3.var()), lat_range),
        var="o3",
        noise="o3_var",
        coords=names,
        epoch=t0,
        mean=ss.mean_basis(ndays, lat_range),
        subsample=1500,
    )
    p = fit.parameters
    print(
        f"fitted in {time.perf_counter() - tic:.0f}s: lengths lat {p['length.lat']:.1f}°, "
        f"lon {p['length.lon']:.0f}°, time {p['length.time']:.1f} d; "
        f"nugget σ {np.sqrt(fit.nugget):.3f}; drift prior {p['mean.1.prior_std']:.3f}"
    )

    # the retrieval errors plus the fitted nugget (variability unresolved by the covariance)
    edges = ss.EDGES
    covered = ss.covered_bands(lat)
    i0, i1 = np.flatnonzero(covered)[[0, -1]]
    est = rb.xarray.optimal_estimate(
        ds,
        {
            "lat": edges[i0 : i1 + 2],
            "lon": [0.0, 360.0],
            "time": [t0, t0 + np.timedelta64(ndays, "D")],
        },
        fit,
        var="o3",
        noise="o3_var",
        coords=names,
        epoch=t0,
        target_weight={"lat": "cos_lat_deg"},
    ).squeeze(["lon", "time"])
    est.to_netcdf(out / f"osiris_{args.year}{args.month:02d}_{args.altitude:g}km.nc")

    g, s = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
    g[i0 : i1 + 1], s[i0 : i1 + 1] = est.o3.values, est.o3_std.values
    y = ds.o3.values
    b = ss.bin_average(y, lat)
    nb = np.array([np.sum((lat >= a) & (lat < c)) for a, c in itertools.pairwise(edges)])
    sem = np.array(
        [
            np.std(y[(lat >= a) & (lat < c)]) / np.sqrt(max(n, 1))
            for (a, c), n in zip(itertools.pairwise(edges), nb, strict=True)
        ]
    )
    print(
        f"\n{'band':>10} {'n':>5} {'bin avg':>8} {'± sem':>6} {'estimate':>9} {'± σ':>6} "
        f"{'est − bin':>9}"
    )
    for i, (a, c) in enumerate(itertools.pairwise(edges)):
        flag = "" if covered[i] else "  (outside coverage)"
        print(
            f"{a:4.0f}..{c:<4.0f} {nb[i]:5d} {b[i]:8.3f} {sem[i]:6.3f} {g[i]:9.3f} {s[i]:6.3f} "
            f"{g[i] - b[i]:9.3f}{flag}"
        )
    np.savez(
        out / f"osiris_{args.year}{args.month:02d}_{args.altitude:g}km.npz",
        edges=edges,
        bin=b,
        est=g,
        std=s,
        n=nb,
        sem=sem,
    )


if __name__ == "__main__":
    main()
