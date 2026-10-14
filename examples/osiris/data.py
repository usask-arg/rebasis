"""Loaders for OSIRIS (HARMOZ) and MLS (L2GP) ozone, and an MLS-derived "true" field.

Paths default to the ARG data volumes; override with the OSIRIS_ROOT and MLS_ROOT environment
variables. Requires h5py (and h5netcdf for xarray): ``uv run --with h5py --with h5netcdf ...``.
"""

from __future__ import annotations

import glob
import itertools
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

OSIRIS_ROOT = os.environ.get(
    "OSIRIS_ROOT", "/Volumes/OsirisDataProducts/OSIRIS/Level2/o3/7.4/HARMOZ"
)
MLS_ROOT = os.environ.get("MLS_ROOT", "/Volumes/root/data_products/MLS/O3/v5")
TAI93 = datetime(1993, 1, 1)
# Files are copied here once: the network shares occasionally fail a read.
CACHE = Path(os.environ.get("REBASIS_DATA_CACHE", Path(__file__).with_name("out") / "cache"))


def local_copy(path: str, attempts: int = 5) -> str:
    """A cached local copy of a file on a network share, retrying failed reads."""
    dest = CACHE / Path(path).name
    if dest.exists():
        return str(dest)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(attempts):
        try:
            # Python's open() has been seen to fail repeatedly on a file of the SMB share that
            # the system cp reads fine, so alternate between them
            if k % 2 == 0:
                shutil.copyfile(path, tmp)
            else:
                subprocess.run(["cp", path, str(tmp)], check=True)
            tmp.rename(dest)
            return str(dest)
        except (OSError, subprocess.CalledProcessError):
            if k == attempts - 1:
                raise
            time.sleep(2.0 * (k + 1))
    raise AssertionError("unreachable")


def month_start(year: int, month: int) -> np.datetime64:
    return np.datetime64(f"{year:04d}-{month:02d}-01")


def days_in_month(year: int, month: int) -> int:
    nxt = datetime(year + month // 12, month % 12 + 1, 1)
    return (nxt - datetime(year, month, 1)).days


def load_osiris(year: int, month: int) -> xr.Dataset:
    """All OSIRIS v7.4 ozone profiles of a month, with ``day`` (fractional days since the start
    of the month) added."""
    (path,) = glob.glob(f"{OSIRIS_ROOT}/*HARMOZ_ALT-{year:04d}{month:02d}-*.nc")
    ds = xr.open_dataset(local_copy(path), engine="h5netcdf").load()
    day = (ds.time.values - month_start(year, month)) / np.timedelta64(1, "D")
    return ds.assign(day=("time", day.astype(float)))


@dataclass
class MLSLevel:
    lat: np.ndarray
    lon: np.ndarray
    day: np.ndarray  # fractional days since the start of the month
    value: np.ndarray  # ppmv
    pressure: float


def load_mls(year: int, month: int, pressure_hpa: float) -> MLSLevel:
    """Screened MLS v5 ozone at the L2GP level nearest ``pressure_hpa``, for a month."""
    start = datetime(year, month, 1)
    lat, lon, day, val = [], [], [], []
    p_used = None
    for d in range(days_in_month(year, month)):
        doy = (start + timedelta(days=d)).timetuple().tm_yday
        files = glob.glob(f"{MLS_ROOT}/MLS-Aura_L2GP-O3_*_{year:04d}d{doy:03d}.he5")
        if not files:
            continue
        with h5py.File(local_copy(files[0]), "r") as f:
            g = f["HDFEOS/SWATHS/O3"]
            p = g["Geolocation Fields/Pressure"][:]
            k = int(np.argmin(np.abs(np.log(p / pressure_hpa))))
            p_used = float(p[k])
            v = g["Data Fields/L2gpValue"][:, k]
            prec = g["Data Fields/L2gpPrecision"][:, k]
            ok = (
                (g["Data Fields/Status"][:] % 2 == 0)
                & (g["Data Fields/Quality"][:] > 1.0)
                & (g["Data Fields/Convergence"][:] < 1.03)
                & (prec > 0)
            )
            t = g["Geolocation Fields/Time"][:]
            when = np.array([TAI93 + timedelta(seconds=float(s)) for s in t[ok]])
            day.append(np.array([(w - start).total_seconds() / 86400 for w in when]))
            lat.append(g["Geolocation Fields/Latitude"][:][ok])
            lon.append(g["Geolocation Fields/Longitude"][:][ok] % 360)
            val.append(v[ok] * 1e6)
    return MLSLevel(*(np.concatenate(a).astype(float) for a in (lat, lon, day, val)), p_used)


class GriddedField:
    """A daily lat × lon grid of bin-averaged MLS values, filled where empty, and its trilinear
    interpolant F(day, lat, lon) (periodic in longitude). F is the "true" field of the sampling
    experiment: it can be sampled anywhere and its exact averages are known."""

    def __init__(self, mls: MLSLevel, ndays: int, dlat: float = 4.0, dlon: float = 30.0):
        self.lat_edges = np.arange(-84.0, 84.0 + 1e-9, dlat)
        self.lon_edges = np.arange(0.0, 360.0 + 1e-9, dlon)
        self.lat_c = 0.5 * (self.lat_edges[1:] + self.lat_edges[:-1])
        self.lon_c = 0.5 * (self.lon_edges[1:] + self.lon_edges[:-1])
        self.day_c = np.arange(ndays) + 0.5
        nlat, nlon = len(self.lat_c), len(self.lon_c)
        grid = np.full((ndays, nlat, nlon), np.nan)
        di = np.floor(mls.day).astype(int)
        li = np.digitize(mls.lat, self.lat_edges) - 1
        oi = np.digitize(mls.lon, self.lon_edges) - 1
        ok = (di >= 0) & (di < ndays) & (li >= 0) & (li < nlat) & (oi >= 0) & (oi < nlon)
        sums = np.zeros_like(grid)
        counts = np.zeros_like(grid)
        np.add.at(sums, (di[ok], li[ok], oi[ok]), mls.value[ok])
        np.add.at(counts, (di[ok], li[ok], oi[ok]), 1)
        with np.errstate(invalid="ignore"):
            grid = sums / counts
        # fill empty cells: periodic interpolation in longitude, then in latitude, then in time
        for d in range(ndays):
            for i in range(nlat):
                row = grid[d, i]
                good = np.isfinite(row)
                if 0 < good.sum() < nlon:
                    x = self.lon_c[good]
                    grid[d, i, ~good] = np.interp(self.lon_c[~good], x, row[good], period=360)
            for j in range(nlon):
                col = grid[d, :, j]
                good = np.isfinite(col)
                if 0 < good.sum() < nlat:
                    grid[d, ~good, j] = np.interp(self.lat_c[~good], self.lat_c[good], col[good])
        for i in range(nlat):
            for j in range(nlon):
                s = grid[:, i, j]
                good = np.isfinite(s)
                if 0 < good.sum() < ndays:
                    grid[~good, i, j] = np.interp(self.day_c[~good], self.day_c[good], s[good])
        if not np.all(np.isfinite(grid)):
            msg = "MLS coverage too sparse to fill the grid"
            raise ValueError(msg)
        self.grid = grid
        self.ndays = ndays

    def __call__(self, day, lat, lon):
        """Trilinear interpolation (clamped at the time and latitude ends, periodic in lon)."""
        day, lat, lon = (np.asarray(a, dtype=float) for a in (day, lat, lon))
        td = np.clip(day - 0.5, 0, self.ndays - 1 - 1e-9)
        t0 = np.floor(td).astype(int)
        ft = td - t0
        tl = np.clip(
            (lat - self.lat_c[0]) / (self.lat_c[1] - self.lat_c[0]), 0, len(self.lat_c) - 1 - 1e-9
        )
        l0 = np.floor(tl).astype(int)
        fl = tl - l0
        dlon = self.lon_c[1] - self.lon_c[0]
        to = ((lon - self.lon_c[0]) % 360) / dlon
        o0 = np.floor(to).astype(int) % len(self.lon_c)
        fo = to - np.floor(to)
        o1 = (o0 + 1) % len(self.lon_c)
        g = self.grid
        t1 = np.minimum(t0 + 1, self.ndays - 1)
        l1 = np.minimum(l0 + 1, len(self.lat_c) - 1)
        out = 0.0
        for wt, ti in ((1 - ft, t0), (ft, t1)):
            for wl, li in ((1 - fl, l0), (fl, l1)):
                for wo, oi in ((1 - fo, o0), (fo, o1)):
                    out = out + wt * wl * wo * g[ti, li, oi]
        return out

    def zonal_monthly_mean(self, edges, nlat_fine: int = 400) -> np.ndarray:
        """The exact-enough area-weighted zonal monthly mean of F in latitude bands (fine
        midpoint quadrature of the interpolant)."""
        out = []
        days = np.linspace(0, self.ndays, 20 * self.ndays + 1)[:-1] + 0.025
        lons = np.arange(0, 360, 1.0) + 0.5
        for a, b in itertools.pairwise(edges):
            lats = a + (np.arange(nlat_fine) + 0.5) * (b - a) / nlat_fine
            dd, ll, oo = np.meshgrid(days, lats, lons, indexing="ij")
            f = self(dd, ll, oo).mean(axis=(0, 2))  # over days and longitudes
            w = np.cos(np.radians(lats))
            out.append(np.sum(f * w) / np.sum(w))
        return np.array(out)
