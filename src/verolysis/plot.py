from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import numpy as np

from verolysis.piecewise_density import PiecewiseDensity


def plot(what, *args, **kwargs):
    if isinstance(what, PiecewiseDensity):
        plot_density(what, *args, **kwargs)


def plot_density(f: PiecewiseDensity, *args, **kwargs) -> None:
    ax = prepare_axis(**kwargs)
    ax.bar(f.va, f.vh, f.vw, align="edge")
    _set_limits(ax, f, **kwargs)


def prepare_axis(**kwargs) -> Axes:
    ax = _get_axis(**kwargs)
    _set_grid(ax, **kwargs)
    _set_xformatter(ax)
    return ax


def _set_limits(ax: Axes, f: PiecewiseDensity, **kwargs) -> None:
    xmin = kwargs.get("xmin", 0)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", 0)
    ymax = kwargs.get("ymax", _guess_height(f))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _set_grid(ax, **kwargs) -> None:
    grid = kwargs.get("grid", True)
    ax.grid(ax)


def _guess_height(f) -> float:
    if len(f) > 100:
        return sorted(f.vh)[-10] * 1.1
    else:
        return np.max(f.vh) * 1.1


def _set_xformatter(ax) -> None:
    ax.get_xaxis().set_major_formatter(FuncFormatter(_xp_repr))


def _get_axis(**kwargs) -> Axes:
    if "ax" in kwargs:
        return kwargs["ax"]
    else:
        figsize = kwargs.get("figsize", None)
        plt.figure(figsize=figsize)
        return plt.gca()


def _xp_repr(x, p):
    if x >= 1_000_000:
        if x % 1_000_000 == 0:
            return f"{int(x / 1_000_000)}M"
        elif x % 100_000 == 0:
            return f"{x / 1_000_000:.1f}M"
        else:
            return f"{x / 1_000_000:.2f}M"
    elif x >= 1_000:
        if x % 1_000 == 0:
            return f"{int(x / 1_000)}k"
        if x % 100 == 0:
            return f"{x / 1_000:.1f}k"
        else:
            return f"{x / 1_000:.2f}k"
    elif x == 0:
        return "0"
    else:
        return str(x)
