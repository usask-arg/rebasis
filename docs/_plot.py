"""Figure style for the documentation pages: the first three slots of a validated categorical
palette, recessive axes and grid."""

import matplotlib as mpl

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#a3a29c"

mpl.rcParams.update(
    {
        "axes.edgecolor": MUTED,
        "axes.labelcolor": "#52514e",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 110,
    }
)
