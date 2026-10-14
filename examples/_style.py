"""Shared figure style: the dataviz reference palette (first three categorical slots,
validated) with recessive axes and grid."""

import matplotlib as mpl

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
SURFACE = "#fcfcfb"


def apply():
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def label(ax, x, y, text, color, **kw):
    """Direct label next to a line, in ink (the colored mark carries identity)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_2,
        fontsize=8,
        va="center",
        **kw,
    )
    ax.plot([x], [y], "o", ms=4, color=color)
