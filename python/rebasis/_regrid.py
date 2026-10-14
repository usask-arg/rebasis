"""Regridding n-dimensional arrays, one axis at a time.

For tensor-product grids and separable weights, applying 1-D maps axis by axis is exactly the
n-dimensional operation (the Gram and cross matrices are Kronecker products), so a lat × lon ×
time regrid is three 1-D regrids.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._ops import Fit, Projection, _bounds, _from_rows, _to_rows, measure
from ._space import Space

METHODS = ("project", "measure", "interpolate", "fit")


def _operator(src: Space, dst: Space, method: str, weight, fit_options: dict[str, Any]):
    """A callable (rows (k, n_src), bounded) -> (k, n_dst)."""
    if method == "project":
        p = Projection(src, dst, weight)
        return lambda rows, bounded: p._inner.apply_last_axis(rows, _bounds(bounded))
    if method == "measure":
        w = measure(dst, src, "integral", weight)
        return lambda rows, bounded: (w @ rows.T).T
    if method == "interpolate":
        e = src.evaluate(dst.centers)
        return lambda rows, bounded: (e @ rows.T).T
    if method == "fit":
        f = Fit(src, dst, weight=weight, **fit_options)
        return lambda rows, bounded: f._inner.apply_last_axis(rows)
    msg = f"method must be one of {METHODS}, got {method!r}"
    raise ValueError(msg)


def regrid(
    data,
    src: Space,
    dst: Space,
    axis: int = -1,
    *,
    method: str = "project",
    weight=None,
    bounded=False,
    skipna: bool = False,
    na_thres: float = 0.5,
    **fit_options,
) -> np.ndarray:
    """Regrids ``data`` (coefficients in ``src`` along ``axis``) to ``dst``.

    Parameters
    ----------
    method
        ``"project"``: L2 projection onto ``dst`` (conservative; the default).
        ``"measure"``: ``dst`` elements are averaging kernels, g_i = ∫ D_i f w / ∫ D_i w.
        ``"interpolate"``: f evaluated at ``dst.centers``.
        ``"fit"``: ``src`` elements are the kernels that measured the data (e.g. bin averages)
        and ``dst`` is a model space; returns the model coefficients (smooth conservative
        upsampling with a spline ``dst``). Extra keyword arguments go to :class:`Fit`.
    weight
        Weight of the inner products, e.g. ``"cos_lat_deg"`` for latitude.
    bounded
        ``"project"`` only: the best conservative approximation within bounds, see
        :meth:`Projection.apply` (``True``/``"local"``: no new extrema; ``"positive"``;
        ``(lo, hi)``).
    skipna
        Treat NaN as missing: regrid the data with NaN set to 0 and divide by the regridded
        validity mask (outputs whose valid fraction is below ``na_thres`` are NaN). Exact for
        maps that preserve constants (projection onto boxes, hats or clamped B-splines, and
        measure).
    """
    if _bounds(bounded) is not None and method != "project":
        msg = "bounded is only supported for method='project'"
        raise ValueError(msg)
    if _bounds(bounded) is not None and skipna:
        msg = "bounded and skipna cannot be combined"
        raise ValueError(msg)
    op = _operator(src, dst, method, weight, fit_options)
    rows, lead = _to_rows(data, axis)
    if rows.shape[1] != len(src):
        msg = f"data has {rows.shape[1]} values along axis {axis}, src has {len(src)} elements"
        raise ValueError(msg)
    if skipna:
        valid = np.isfinite(rows)
        filled = np.ascontiguousarray(np.where(valid, rows, 0.0))
        num = op(filled, None)
        den = op(np.ascontiguousarray(valid.astype(np.float64)), None)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den >= na_thres, num / den, np.nan)
    else:
        out = op(rows, bounded)
    return _from_rows(out, lead, axis)


def regrid_nd(
    data,
    axes: Mapping[int, tuple],
    *,
    method: str = "project",
    bounded=False,
    skipna: bool = False,
    na_thres: float = 0.5,
    **fit_options,
) -> np.ndarray:
    """Regrids several axes: ``axes`` maps axis -> (src, dst) or (src, dst, weight)."""
    out = np.asarray(data, dtype=np.float64)
    for axis, spec in axes.items():
        src, dst, *rest = spec
        weight = rest[0] if rest else None
        out = regrid(
            out,
            src,
            dst,
            axis,
            method=method,
            weight=weight,
            bounded=bounded,
            skipna=skipna,
            na_thres=na_thres,
            **fit_options,
        )
    return out
