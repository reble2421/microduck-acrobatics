"""Shared look of the report figures (matplotlib): font, colours, axes.

Colours are the validated reference palette (categorical slots in fixed order, light
surface), text stays in ink colours, the grid is recessive. Import apply() first.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT = "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc"     # BIZ UDGothic (SIL Open Font License)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (light surface)
INK, INK2, MUTED, GRID, BAND, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#efeeea", "#fcfcfb"
DPI = 200


def apply():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(), "font.size": 11,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titlelocation": "left",
        "legend.frameon": False, "legend.labelcolor": INK, "lines.linewidth": 2.0,
        "savefig.facecolor": SURFACE,
    })


def grid_y(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def wilson(k, n, z=1.96):
    """95 % Wilson interval of k successes in n trials, as (low, high) fractions."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)
