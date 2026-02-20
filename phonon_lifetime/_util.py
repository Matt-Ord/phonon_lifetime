from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d.axes3d import Axes3D


def get_axis(ax: Axes | None) -> tuple[Figure, Axes]:
    """Get the axis to plot on."""
    if ax is None:
        fig, ax = plt.subplots()
        assert ax is not None
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax


def get_axis_3d(ax: Axes3D | None) -> tuple[Figure, Axes3D]:
    """Get a 3D axis to plot on."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax
