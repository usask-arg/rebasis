"""Sampling error of OSIRIS monthly zonal means: bin averages vs rebasis optimal estimates.

MLS, which samples densely and evenly, gives a "true" field F (daily lat × lon grid at one
pressure level). F is sampled at OSIRIS's actual locations and times; the monthly zonal means
are then computed from those samples by bin averaging and by ``rebasis.optimal_estimate``, and
compared with F's exact monthly zonal means. The differences are pure sampling error (Toohey
et al. 2013, Sofieva et al. 2014).

    uv run --with h5py --with h5netcdf python examples/osiris/sampling_study.py 2010 9 46
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from data import GriddedField, days_in_month, load_mls, load_osiris
from scipy import optimize

import rebasis as rb

EDGES = np.arange(-80.0, 80.0 + 1e-9, 10.0)


LAT_KNOT_SPACING = 30.0
TWO_SCALE = False
LAT_PROFILE = False
DRIFT_PRIOR = None  # ppmv over the month; None: an unconstrained linear drift


def mean_basis(ndays, lat_range=(-90.0, 90.0)):
    """A smooth latitude structure (cubic B-splines) drifting linearly through the month.

    With DRIFT_PRIOR set, the latitude structure is unconstrained and the drift (latitude
    structure × a centred linear function of time) has that prior standard deviation over the
    month, so where only part of the month was sampled the drift shrinks towards zero instead
    of being extrapolated."""
    lo, hi = lat_range
    n = max(2, round((hi - lo) / LAT_KNOT_SPACING) + 1)
    lat = rb.Space.bsplines_clamped(np.linspace(lo, hi, n), 3)
    if DRIFT_PRIOR is None:
        return {"lat": lat, "time": rb.Space.hats([0.0, float(ndays)])}
    t = float(ndays)
    drift = rb.Space.shifted([t / 2], [-t / 2, t / 2], [-0.5, 0.5])  # (time − t/2) / t
    return [
        rb.MeanTerm({"lat": lat}),
        rb.MeanTerm({"lat": lat, "time": drift}, prior_std=DRIFT_PRIOR),
    ]


def _component(var, l_lat, l_lon, l_time):
    return rb.Covariance(
        var,
        lat=rb.CovarianceDim("squared_exponential", l_lat),
        lon=rb.CovarianceDim("exponential", l_lon, period=360.0),
        time=rb.CovarianceDim("matern32", l_time),
    )


def covariance(theta):
    """θ = log of (variance, lengths in lat, lon, time) per component, then the noise variance:
    5 parameters for one component, 9 for two (small scale + large scale)."""
    p = np.exp(theta)
    cov = _component(*p[:4])
    if len(theta) == 9:
        cov = cov + _component(*p[4:8])
    return cov


def fit_covariance(y, coords, ndays, n_sub=1500, seed=0, lat_range=(-90.0, 90.0)):
    """Maximize the (REML) likelihood over variance, three lengths and the noise, on a
    subsample."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), min(n_sub, len(y)), replace=False)
    c = {k: np.asarray(v)[sub] for k, v in coords.items()}
    ys = y[sub]
    mb = mean_basis(ndays, lat_range)

    def nll(theta):
        try:
            return -rb.log_likelihood(ys, c, covariance(theta), noise=np.exp(theta[-1]), mean=mb)
        except rb.SingularError:
            return 1e30

    v = np.var(ys)
    if TWO_SCALE:
        # a small-scale and a large-scale component, with non-overlapping length ranges
        theta0 = np.log([0.5 * v, 5.0, 40.0, 2.0, 0.5 * v, 20.0, 150.0, 15.0, 0.01 * v])
        bounds = [
            (np.log(1e-3 * v), np.log(10 * v)),
            (np.log(1.0), np.log(15.0)),
            (np.log(5.0), np.log(90.0)),
            (np.log(0.3), np.log(5.0)),
            (np.log(1e-3 * v), np.log(10 * v)),
            (np.log(8.0), np.log(90.0)),
            (np.log(90.0), np.log(720.0)),
            (np.log(5.0), np.log(90.0)),
            (np.log(1e-4 * v), np.log(v)),
        ]
    else:
        theta0 = np.log([v, 10.0, 60.0, 5.0, 0.01 * v])
        bounds = [
            (np.log(1e-3 * v), np.log(10 * v)),
            (np.log(2.0), np.log(60.0)),
            (np.log(10.0), np.log(360.0)),
            (np.log(0.5), np.log(60.0)),
            (np.log(1e-4 * v), np.log(v)),
        ]
    res = optimize.minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
    return res.x, -res.fun


def fit_with_rebasis(y, coords, ndays, lat_range, n_sub=1500):
    """rb.fit_covariance with a fitted drift prior: lengths, variance, nugget and the width of
    the drift, all by maximum likelihood with analytic gradients."""
    global DRIFT_PRIOR
    DRIFT_PRIOR = DRIFT_PRIOR or 0.1
    v = float(np.var(y))
    start = _component(v, 10.0, 60.0, 5.0)
    if LAT_PROFILE:
        # the standard deviation as a smooth function of latitude (the variance is then 1)
        lo, hi = lat_range
        basis = rb.Space.bsplines_clamped(np.linspace(lo, hi, round((hi - lo) / 30) + 1), 3)
        profile = rb.VarianceProfile(basis, np.full(len(basis), np.sqrt(v)))
        start = rb.Covariance(
            1.0,
            lat=rb.CovarianceDim("squared_exponential", 10.0, variance_profile=profile),
            lon=rb.CovarianceDim("exponential", 60.0, period=360.0),
            time=rb.CovarianceDim("matern32", 5.0),
        )
    return rb.fit_covariance(
        y,
        coords,
        start,
        noise=0.0,
        mean=mean_basis(ndays, lat_range),
        nugget=0.01 * v,
        subsample=n_sub,
    )


def describe(theta):
    p = np.exp(theta)
    parts = [
        f"σ {np.sqrt(p[i]):.3f} (lat {p[i + 1]:.1f}°, lon {p[i + 2]:.0f}°, t {p[i + 3]:.1f} d)"
        for i in range(0, len(theta) - 1, 4)
    ]
    return " + ".join(parts) + f", noise σ {np.sqrt(p[-1]):.3f}"


def coverage(lat):
    """The latitude range of the data, rounded outward to 5°: the mean basis spans only this
    (its coefficients beyond the data would be undetermined), and bands outside it are not
    estimated."""
    return max(-90.0, 5 * np.floor(lat.min() / 5)), min(90.0, 5 * np.ceil(lat.max() / 5))


def covered_bands(lat):
    lo, hi = coverage(lat)
    return (EDGES[:-1] >= lo) & (EDGES[1:] <= hi)


def bin_average(y, lat):
    return np.array(
        [
            y[(lat >= a) & (lat < b)].mean() if np.any((lat >= a) & (lat < b)) else np.nan
            for a, b in itertools.pairwise(EDGES)
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("month", type=int)
    ap.add_argument("pressure", type=float, help="hPa")
    ap.add_argument("--out", default=str(Path(__file__).with_name("out")))
    ap.add_argument("--knots", type=float, default=30.0, help="latitude knot spacing of the mean")
    ap.add_argument(
        "--fit-prior",
        action="store_true",
        help="fit everything, the drift prior included, with rebasis.fit_covariance",
    )
    ap.add_argument(
        "--lat-profile",
        action="store_true",
        help="with --fit-prior: a fitted variance profile in latitude",
    )
    ap.add_argument(
        "--drift-prior",
        type=float,
        default=None,
        help="prior std of the monthly drift (default: unconstrained)",
    )
    ap.add_argument(
        "--two-scale",
        action="store_true",
        help="a small-scale plus a large-scale covariance component",
    )
    ap.add_argument(
        "--local",
        type=float,
        default=0.0,
        help="estimate each band from observations within this many degrees of it, "
        "with its own fitted covariance (0: one global fit)",
    )
    args = ap.parse_args()
    global LAT_KNOT_SPACING, TWO_SCALE, DRIFT_PRIOR, LAT_PROFILE
    LAT_KNOT_SPACING = args.knots
    TWO_SCALE = args.two_scale
    DRIFT_PRIOR = args.drift_prior
    LAT_PROFILE = args.lat_profile
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ndays = days_in_month(args.year, args.month)

    t0 = time.perf_counter()
    mls = load_mls(args.year, args.month, args.pressure)
    field = GriddedField(mls, ndays)
    truth = field.zonal_monthly_mean(EDGES)
    osi = load_osiris(args.year, args.month)
    lat, lon, day = (
        osi.latitude.values.astype(float),
        osi.longitude.values.astype(float) % 360,
        osi.day.values,
    )
    y = field(day, lat, lon)  # F sampled like OSIRIS
    coords = {"lat": lat, "lon": lon, "time": day}
    print(
        f"loaded {len(y)} OSIRIS samples of MLS at {mls.pressure:.1f} hPa "
        f"in {time.perf_counter() - t0:.0f}s"
    )

    if args.local > 0:
        run_local(args, y, coords, lat, truth, ndays, out)
        return

    t0 = time.perf_counter()
    if args.fit_prior:
        fit = fit_with_rebasis(y, coords, ndays, coverage(lat))
        cov_model, noise, mean_model = fit.covariance, fit.nugget, fit.mean
        p = fit.parameters
        print(
            f"fitted with rebasis in {time.perf_counter() - t0:.0f}s ({fit.n_iterations} "
            f"iterations): σ {np.sqrt(p['variance']):.3f} (lat {p['length.lat']:.1f}°, lon "
            f"{p['length.lon']:.0f}°, t {p['length.time']:.1f} d), drift prior "
            f"{p['mean.1.prior_std']:.3f}, nugget σ {np.sqrt(fit.nugget):.3f}"
        )
        if LAT_PROFILE:
            s = fit.covariance.dims["lat"].variance_profile
            grid = np.arange(-75.0, 80.0, 15.0)
            grid = grid[(grid >= coverage(lat)[0]) & (grid <= coverage(lat)[1])]
            print(
                "  fitted σ(lat): "
                + ", ".join(f"{g:.0f}°: {v:.2f}" for g, v in zip(grid, s(grid), strict=True))
            )
    else:
        theta, _ll = fit_covariance(y, coords, ndays, lat_range=coverage(lat))
        noise = float(np.exp(theta[-1]))
        cov_model, mean_model = covariance(theta), mean_basis(ndays, coverage(lat))
        print(f"fitted in {time.perf_counter() - t0:.0f}s: {describe(theta)}")

    t0 = time.perf_counter()
    # covered bands are contiguous: estimate those, report the others as not covered
    covered = covered_bands(lat)
    i0, i1 = np.flatnonzero(covered)[[0, -1]]
    targets = {
        "lat": rb.Space.boxes(EDGES[i0 : i1 + 2]),
        "lon": rb.Space.boxes([0, 360]),
        "time": rb.Space.boxes([0, ndays]),
    }
    est = rb.optimal_estimate(
        y,
        coords,
        targets,
        cov_model,
        noise=noise,
        target_weight={"lat": "cos_lat_deg"},
        mean=mean_model,
    )
    print(f"estimated in {time.perf_counter() - t0:.0f}s")
    g, s = np.full(len(truth), np.nan), np.full(len(truth), np.nan)
    g[i0 : i1 + 1], s[i0 : i1 + 1] = est.values[:, 0, 0], est.std[:, 0, 0]
    b = np.where(covered, bin_average(y, lat), np.nan)

    print(f"\n{'band':>10} {'truth':>7} {'bin err':>8} {'est err':>8} {'2σ':>6}  {'|err|/σ':>7}")
    for a, tr, bi, gi, si in zip(EDGES[:-1], truth, b, g, s, strict=True):
        if np.isnan(gi):
            print(f"{a:4.0f}..{a + 10:<4.0f} {tr:7.3f}   (not covered by OSIRIS this month)")
            continue
        print(
            f"{a:4.0f}..{a + 10:<4.0f} {tr:7.3f} {bi - tr:8.3f} {gi - tr:8.3f} {2 * si:6.3f}"
            f"  {abs(gi - tr) / si:7.2f}"
        )
    rms = lambda e: np.sqrt(np.nanmean(e**2))  # noqa: E731
    print(
        f"\nRMS sampling error: bin average {rms(b - truth):.3f}, optimal estimate "
        f"{rms(g - truth):.3f} ppmv; truth within 2σ in "
        f"{np.sum(np.abs(g - truth) < 2 * s)}/{np.sum(covered)} covered bands"
    )
    model = (
        "lat_profile"
        if LAT_PROFILE
        else "fitted_prior"
        if args.fit_prior
        else "two_scale"
        if TWO_SCALE
        else "global"
    )
    np.savez(
        out / f"sampling_{model}_{args.year}{args.month:02d}_{args.pressure:g}hPa.npz",
        edges=EDGES,
        truth=truth,
        bin=b,
        est=g,
        std=s,
        lat=lat,
        lon=lon,
        day=day,
        y=y,
        weights=est.weights,
    )


def run_local(args, y, coords, lat, truth, ndays, out):
    g, s = np.full(len(truth), np.nan), np.full(len(truth), np.nan)
    print(
        f"\n{'band':>10} {'n':>5} {'σ':>6} {'ℓ_lat':>6} {'ℓ_lon':>6} {'ℓ_t':>5} "
        f"{'truth':>7} {'bin err':>8} {'est err':>8} {'2σ':>6} {'|err|/σ':>7}"
    )
    b = bin_average(y, lat)
    covered = covered_bands(lat)
    b = np.where(covered, b, np.nan)
    cov_lo, cov_hi = coverage(lat)
    for i, (a, c) in enumerate(itertools.pairwise(EDGES)):
        if not covered[i]:
            continue
        lo, hi = max(a - args.local, cov_lo), min(c + args.local, cov_hi)
        sel = (lat >= lo) & (lat < hi)
        cs = {k: np.asarray(v)[sel] for k, v in coords.items()}
        theta, _ = fit_covariance(y[sel], cs, ndays, lat_range=(lo, hi))
        var, l_lat, l_lon, l_time = np.exp(theta[:4])
        noise = float(np.exp(theta[-1]))
        targets = {
            "lat": rb.Space.boxes([a, c]),
            "lon": rb.Space.boxes([0, 360]),
            "time": rb.Space.boxes([0, ndays]),
        }
        est = rb.optimal_estimate(
            y[sel],
            cs,
            targets,
            covariance(theta),
            noise=noise,
            target_weight={"lat": "cos_lat_deg"},
            mean=mean_basis(ndays, (lo, hi)),
        )
        g[i], s[i] = est.values.ravel()[0], est.std.ravel()[0]
        print(
            f"{a:4.0f}..{c:<4.0f} {sel.sum():5d} {np.sqrt(var):6.3f} {l_lat:6.1f} {l_lon:6.0f} "
            f"{l_time:5.1f} {truth[i]:7.3f} {b[i] - truth[i]:8.3f} {g[i] - truth[i]:8.3f} "
            f"{2 * s[i]:6.3f} {abs(g[i] - truth[i]) / s[i]:7.2f}"
        )
    rms = lambda e: np.sqrt(np.nanmean(e**2))  # noqa: E731
    print(
        f"\nRMS sampling error: bin average {rms(b - truth):.3f}, optimal estimate "
        f"{rms(g - truth):.3f} ppmv; truth within 2σ in "
        f"{np.sum(np.abs(g - truth) < 2 * s)}/{np.sum(covered)} covered bands"
    )
    np.savez(
        out / f"sampling_local_{args.year}{args.month:02d}_{args.pressure:g}hPa.npz",
        edges=EDGES,
        truth=truth,
        bin=b,
        est=g,
        std=s,
    )


if __name__ == "__main__":
    main()
