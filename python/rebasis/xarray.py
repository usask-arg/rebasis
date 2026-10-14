"""xarray front-ends (requires xarray): regridding along named dimensions, and optimal
estimation from observations stored along one dimension of a Dataset or DataArray."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._estimate import optimal_estimate as _optimal_estimate
from ._fit import CovarianceFit
from ._fit import fit_covariance as _fit_covariance
from ._regrid import regrid as _regrid
from ._space import Space

__all__ = ["fit_covariance", "numeric_time", "optimal_estimate", "regrid"]


def _space(values, kind: str) -> Space:
    if kind == "boxes":
        return Space.boxes_from_centers(values)
    if kind == "hats":
        return Space.hats(values)
    msg = f"kind must be 'boxes' (cell averages) or 'hats' (point samples), got {kind!r}"
    raise ValueError(msg)


def regrid(
    obj,
    dst: Mapping[str, Any],
    *,
    src: Mapping[str, Space] | None = None,
    src_kind: str = "boxes",
    dst_kind: str = "boxes",
    weight: Mapping[str, Any] | None = None,
    method: str = "project",
    bounded=False,
    skipna: bool = False,
    na_thres: float = 0.5,
    **fit_options,
):
    """Regrids a DataArray or Dataset along the dimensions in ``dst``.

    Parameters
    ----------
    dst
        dim -> target :class:`Space`, or the target coordinate values (built into a space with
        ``dst_kind``). The new coordinate is the target space's centers.
    src
        dim -> source :class:`Space`. Missing dims are built from the existing coordinate with
        ``src_kind``: ``"boxes"`` (values are cell averages, edges halfway between
        coordinates) or ``"hats"`` (values are point samples). A decreasing coordinate is
        sorted first.
    weight
        dim -> weight, e.g. ``{"lat": "cos_lat_deg"}`` for area-weighted regridding.
    method, bounded, skipna, na_thres, fit_options
        As in :func:`rebasis.regrid`.

    Data variables without any of the regridded dimensions are left untouched; other
    coordinates along a regridded dimension are dropped. The computation is eager.
    """
    import xarray as xr

    src = dict(src or {})
    weight = dict(weight or {})
    for dim in dst:
        if dim not in obj.dims:
            msg = f"{dim!r} is not a dimension of the object"
            raise ValueError(msg)
        if dim not in src and dim in obj.coords:
            c = obj[dim].values
            if c.size > 1 and c[0] > c[-1]:
                obj = obj.sortby(dim)

    def one(da: xr.DataArray) -> xr.DataArray:
        out = da
        for dim, target in dst.items():
            if dim not in out.dims:
                continue
            s = src.get(dim)
            if s is None:
                if dim not in out.coords:
                    msg = f"no coordinate for {dim!r}: pass src={{{dim!r}: Space}}"
                    raise ValueError(msg)
                s = _space(out[dim].values, src_kind)
            d = target if isinstance(target, Space) else _space(np.asarray(target), dst_kind)
            values = _regrid(
                out.values,
                s,
                d,
                out.get_axis_num(dim),
                method=method,
                weight=weight.get(dim),
                bounded=bounded,
                skipna=skipna,
                na_thres=na_thres,
                **fit_options,
            )
            coords = {k: v for k, v in out.coords.items() if dim not in v.dims}
            coords[dim] = d.centers
            out = xr.DataArray(values, dims=out.dims, coords=coords, attrs=out.attrs, name=out.name)
        return out

    if isinstance(obj, xr.DataArray):
        return one(obj)
    if isinstance(obj, xr.Dataset):
        regridded = {k: one(v) for k, v in obj.data_vars.items() if any(d in v.dims for d in dst)}
        kept = obj.drop_vars(list(regridded)).drop_dims(
            [d for d in dst if d in obj.dims], errors="ignore"
        )
        return xr.merge([kept, xr.Dataset(regridded)], combine_attrs="override")
    msg = f"expected an xarray DataArray or Dataset, got {type(obj).__name__}"
    raise TypeError(msg)


# ------------------------------------------------------------------------------- estimation


def numeric_time(times, epoch="1970-01-01", unit: str = "D") -> np.ndarray:
    """Datetimes as floats: the time since ``epoch`` in units of ``unit`` (a numpy timedelta
    unit, e.g. ``"D"`` for days or ``"h"`` for hours).

    :func:`optimal_estimate` and :func:`fit_covariance` convert datetime coordinates this way,
    so covariance lengths, periods, neighbourhood widths and mean bases along a time dimension
    are in these units; use this function to build such a basis, e.g.
    ``Space.hats(numeric_time(month_edges))``.
    """
    t = np.asarray(times)
    if not np.issubdtype(t.dtype, np.datetime64):
        msg = f"expected datetime64 values, got {t.dtype}"
        raise TypeError(msg)
    return (t - np.datetime64(epoch)) / np.timedelta64(1, unit)


_CF_UNITS = {"D": "days", "h": "hours", "m": "minutes", "s": "seconds"}


def _from_numeric(x, epoch, unit: str) -> np.ndarray:
    ns = np.round(np.asarray(x) * (np.timedelta64(1, unit) / np.timedelta64(1, "ns")))
    return np.datetime64(epoch, "ns") + ns.astype("timedelta64[ns]")


@dataclass
class _Observations:
    values: Any  # the DataArray of values, (obs_dim, *extra)
    y: np.ndarray  # (n,) or (n, k)
    noise: Any
    coords: dict
    obs_dim: str
    extra: tuple[str, ...]
    times: set[str]  # covariance dimensions given as datetimes


def _observations(obj, names, noise, coords, obs_dim, var, epoch, unit) -> _Observations:
    import xarray as xr

    if isinstance(obj, xr.Dataset):
        if var is None:
            msg = "pass var=<name of the data variable> with a Dataset"
            raise ValueError(msg)
        da = obj[var]
    elif isinstance(obj, xr.DataArray):
        if var is not None:
            msg = "var is only for Datasets"
            raise ValueError(msg)
        da = obj
    else:
        msg = f"expected an xarray DataArray or Dataset, got {type(obj).__name__}"
        raise TypeError(msg)
    coords = dict(coords or {})
    unknown = set(coords) - set(names)
    if unknown:
        msg = f"coords has entries {sorted(unknown)} that are not covariance dimensions {names}"
        raise ValueError(msg)

    sources = {}
    for name in names:
        src = coords.get(name, name)
        if isinstance(src, str):
            if src not in obj.coords and not (isinstance(obj, xr.Dataset) and src in obj):
                msg = (
                    f"no variable or coordinate {src!r} for the covariance dimension {name!r}: "
                    f"pass coords={{{name!r}: <name>}}"
                )
                raise ValueError(msg)
            src = obj[src]
        sources[name] = src
    if obs_dim is None:
        dims = {s.dims for s in sources.values() if isinstance(s, xr.DataArray)}
        if len(dims) != 1 or len(next(iter(dims))) != 1:
            msg = (
                "cannot infer the observation dimension from the coordinates "
                f"(dimensions {sorted(dims)}): pass obs_dim"
            )
            raise ValueError(msg)
        (obs_dim,) = next(iter(dims))
    if obs_dim not in da.dims:
        msg = f"the values have no dimension {obs_dim!r}"
        raise ValueError(msg)
    n = da.sizes[obs_dim]

    out, times = {}, set()
    for name, src in sources.items():
        if isinstance(src, Space):
            out[name] = src
            continue
        if isinstance(src, xr.DataArray):
            if src.dims != (obs_dim,):
                msg = f"the coordinate for {name!r} must lie along {obs_dim!r}, not {src.dims}"
                raise ValueError(msg)
            src = src.values
        src = np.asarray(src)
        if np.issubdtype(src.dtype, np.datetime64):
            src = numeric_time(src, epoch, unit)
            times.add(name)
        if src.shape != (n,):
            msg = f"the coordinate for {name!r} has shape {src.shape}, expected ({n},)"
            raise ValueError(msg)
        out[name] = src.astype(np.float64)

    extra = tuple(d for d in da.dims if d != obs_dim)
    da = da.transpose(obs_dim, *extra)
    y = da.values.astype(np.float64).reshape(n, -1) if extra else da.values.astype(np.float64)
    if isinstance(noise, str):
        noise = obj[noise]
    if isinstance(noise, xr.DataArray):
        missing = set(noise.dims) - set(da.dims)
        if missing:
            msg = f"the noise has dimensions {sorted(missing)} that the values do not"
            raise ValueError(msg)
        if extra:
            noise = noise.broadcast_like(da).transpose(obs_dim, *extra).values.reshape(n, -1)
        else:
            noise = noise.broadcast_like(da).values
        noise = noise.astype(np.float64)
    return _Observations(da, y, noise, out, obs_dim, extra, times)


def _target_spaces(targets, names, times, epoch, unit) -> dict[str, Space]:
    if set(targets) != set(names):
        msg = f"targets must have exactly the covariance's dimensions {names}"
        raise ValueError(msg)
    out = {}
    for name in names:
        t = targets[name]
        if not isinstance(t, Space):
            t = np.asarray(t)
            if np.issubdtype(t.dtype, np.datetime64):
                t = numeric_time(t, epoch, unit)
            elif name in times:
                msg = f"the {name!r} coordinates are datetimes, so its target edges must be too"
                raise TypeError(msg)
            t = Space.boxes(t)
        out[name] = t
    return out


def _resolve(covariance, kwargs):
    """A CovarianceFit supplies the covariance, its mean (unless given) and its nugget."""
    if isinstance(covariance, CovarianceFit):
        kwargs.setdefault("mean", covariance.mean)
        return covariance.covariance, covariance.nugget
    return covariance, 0.0


def optimal_estimate(
    obj,
    targets: Mapping[str, Any],
    covariance,
    *,
    noise,
    coords: Mapping[str, Any] | None = None,
    obs_dim: str | None = None,
    var: str | None = None,
    epoch="1970-01-01",
    time_unit: str = "D",
    return_estimate: bool = False,
    **kwargs,
):
    """:func:`rebasis.optimal_estimate` for observations stored along one dimension of an xarray
    object, e.g. a Dataset of satellite profiles ``(time, altitude)`` with latitude and
    longitude variables along ``time``.

    Parameters
    ----------
    obj
        A DataArray of values, or a Dataset with the values in variable ``var``. Every dimension
        of the values other than ``obs_dim`` (e.g. altitude) is estimated independently, in one
        call: it becomes a column of values.
    targets
        For each dimension of the covariance: a :class:`Space`, or the edges of boxes (bin
        averages), as datetimes for a datetime coordinate.
    covariance
        A :class:`Covariance` or :class:`CovarianceSum`, or a :class:`CovarianceFit`, whose
        covariance, mean (unless ``mean`` is given) and nugget are used.
    noise
        Observation error variances: a scalar, a DataArray (broadcast against the values), or
        the name of a variable or coordinate of ``obj``.
    coords
        Covariance dimension -> the name of the variable or coordinate holding the observations'
        coordinate (by default the dimension's own name), or the coordinate values, or a
        :class:`Space` of n elements (observations that are averages, e.g. over a footprint).
    obs_dim
        The dimension along which the observations lie; inferred from the coordinates.
    epoch, time_unit
        Datetime coordinates and targets become floats with :func:`numeric_time`: covariance
        lengths, periods, neighbourhood widths and mean bases along them are in these units
        (days by default).
    return_estimate
        Also return the :class:`Estimate` (for its weights, ``apply`` and
        ``weight_histogram``).
    kwargs
        Passed to :func:`rebasis.optimal_estimate`: ``mean``, ``target_weight``,
        ``neighborhood``, ``weights``, ``full_covariance``.

    Returns
    -------
    xarray.Dataset
        With the covariance's dimensions (coordinates: the target centers, and their supports
        as ``<dim>_bounds``) followed by the other dimensions of the values, and variables

        - ``<name>`` (the values' name, or ``"estimate"``): the estimates;
        - ``<name>_std``: their posterior standard deviations;
        - ``prior_std``: the prior standard deviations of the targets;
        - ``n_used``: the number of valid observations each estimate used;
        - ``log_likelihood``, and ``constant_mean`` when the mean is a single estimated or
          known constant: one per column.
    """
    import xarray as xr

    covariance, nugget = _resolve(covariance, kwargs)
    names = covariance.names
    o = _observations(obj, names, noise, coords, obs_dim, var, epoch, time_unit)
    spaces = _target_spaces(targets, names, o.times, epoch, time_unit)
    noise = np.asarray(o.noise, dtype=np.float64) + nugget
    est = _optimal_estimate(o.y, o.coords, spaces, covariance, noise=noise, **kwargs)

    da = o.values
    extra_shape = tuple(da.sizes[d] for d in o.extra)
    grid = tuple(len(spaces[n]) for n in names)

    def full(a):
        return np.asarray(a).reshape(*grid, *extra_shape)

    def per_column(a):
        return (o.extra, np.asarray(a, dtype=np.float64).reshape(extra_shape))

    out_coords = {}
    for name in names:
        s = spaces[name]
        c, lo, hi = s.centers, s.lower, s.upper
        if name in o.times:
            c, lo, hi = (_from_numeric(a, epoch, time_unit) for a in (c, lo, hi))
        out_coords[name] = c
        out_coords[f"{name}_bounds"] = ((name, "bounds"), np.stack([lo, hi], axis=-1))
    for k, v in da.coords.items():
        if v.dims and set(v.dims) <= set(o.extra):
            out_coords[k] = v

    label = da.name if isinstance(da.name, str) else "estimate"
    dims = (*names, *o.extra)
    data = {
        label: (dims, full(est.values), dict(da.attrs)),
        f"{label}_std": (dims, full(est.std), {"long_name": "posterior standard deviation"}),
        "prior_std": (names, est.prior_std, {"long_name": "prior standard deviation"}),
        "n_used": (dims, full(est.n_used), {"long_name": "number of observations used"}),
        "log_likelihood": per_column(est.log_likelihood),
    }
    if np.all(np.isfinite(est.mean)):
        data["constant_mean"] = per_column(np.broadcast_to(est.mean, extra_shape))
    ds = xr.Dataset(data, coords=out_coords)
    for name in names:
        ds[name].attrs["bounds"] = f"{name}_bounds"
        if name in o.times and time_unit in _CF_UNITS:
            # CF: a datetime coordinate and its bounds encoded in the same units
            units = f"{_CF_UNITS[time_unit]} since {np.datetime64(epoch, 's')}"
            for v in (name, f"{name}_bounds"):
                ds[v].encoding.update(units=units, dtype="float64")
    return (ds, est) if return_estimate else ds


def fit_covariance(
    obj,
    covariance,
    *,
    noise,
    coords: Mapping[str, Any] | None = None,
    obs_dim: str | None = None,
    var: str | None = None,
    epoch="1970-01-01",
    time_unit: str = "D",
    **kwargs,
) -> CovarianceFit:
    """:func:`rebasis.fit_covariance` for observations stored along one dimension of an xarray
    object. The values must have only that dimension (select e.g. one altitude first).

    ``obj``, ``noise``, ``coords``, ``obs_dim``, ``var``, ``epoch`` and ``time_unit`` are as in
    :func:`optimal_estimate`; ``kwargs`` are passed to :func:`rebasis.fit_covariance`. The
    result can be passed as the covariance of :func:`optimal_estimate`.
    """
    names = covariance.names
    o = _observations(obj, names, noise, coords, obs_dim, var, epoch, time_unit)
    if o.extra:
        msg = (
            f"fit_covariance takes one column of values, but they also have dimensions "
            f"{o.extra}: select one first (e.g. .sel(altitude=...))"
        )
        raise ValueError(msg)
    return _fit_covariance(o.y, o.coords, covariance, noise=o.noise, **kwargs)
