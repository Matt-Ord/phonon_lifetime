from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def get_axis(ax: Axes | None) -> tuple[Figure, Axes]:
    """Get the axis to plot on."""
    if ax is None:
        fig, ax = plt.subplots()
        assert ax is not None
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax
