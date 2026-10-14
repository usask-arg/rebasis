"""Regridding xarray objects along named dimensions (requires xarray)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._regrid import regrid as _regrid
from ._space import Space


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
