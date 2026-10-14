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
from data import days_in_month, load_osiris

import rebasis as rb

AVOGADRO = 6.02214076e23


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("month", type=int)
    ap.add_argument("altitude", type=float, help="km")
    ap.add_argument("--knots", type=float, default=15.0)
    ap.add_argument("--two-scale", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("out")))
    args = ap.parse_args()
    ss.LAT_KNOT_SPACING = args.knots
    ss.TWO_SCALE = args.two_scale
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ndays = days_in_month(args.year, args.month)

    ds = load_osiris(args.year, args.month).sel(altitude=args.altitude, method="nearest")
    # number density in 10^12 cm^-3
    scale = AVOGADRO * 1e-6 / 1e12
    y = ds.ozone_concentration.values * scale
    err = ds.ozone_concentration_standard_error.values * scale
    ok = np.isfinite(y) & np.isfinite(err) & (err > 0)
    lat = ds.latitude.values[ok].astype(float)
    lon = ds.longitude.values[ok].astype(float) % 360
    day = ds.day.values[ok]
    y, err = y[ok], err[ok]
    coords = {"lat": lat, "lon": lon, "time": day}
    print(f"{ok.sum()} OSIRIS profiles at {float(ds.altitude):.0f} km")

    t0 = time.perf_counter()
    theta, _ = ss.fit_covariance(y, coords, ndays, lat_range=ss.coverage(lat))
    nugget = float(np.exp(theta[-1]))
    print(f"fitted in {time.perf_counter() - t0:.0f}s: {ss.describe(theta)}")
    # the retrieval errors plus the fitted nugget (variability unresolved by the covariance)
    noise = err**2 + nugget
    edges = ss.EDGES
    covered = ss.covered_bands(lat)
    i0, i1 = np.flatnonzero(covered)[[0, -1]]
    targets = {
        "lat": rb.Space.boxes(edges[i0 : i1 + 2]),
        "lon": rb.Space.boxes([0, 360]),
        "time": rb.Space.boxes([0, ndays]),
    }
    est = rb.optimal_estimate(
        y,
        coords,
        targets,
        ss.covariance(theta),
        noise=noise,
        target_weight={"lat": "cos_lat_deg"},
        mean=ss.mean_basis(ndays, ss.coverage(lat)),
    )
    g, s = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
    g[i0 : i1 + 1], s[i0 : i1 + 1] = est.values[:, 0, 0], est.std[:, 0, 0]
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
