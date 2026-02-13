from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.animation import ArtistAnimation

from phonon_lifetime._util import get_axis
from phonon_lifetime.modes._util import get_mode_displacement
from phonon_lifetime.system import get_atom_centres

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.quiver import Quiver

    from phonon_lifetime.modes._mode import NormalMode


def plot_mode_1d_x(
    mode: NormalMode,
    time: float = 0,
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = get_axis(ax)

    displacement = get_mode_displacement(mode, time=time)
    displacement = displacement.reshape(*mode.system.n_repeats, 3)
    displacement_x = displacement[:, *idx, 0]
    displacement_x = np.append(displacement_x, displacement_x[0])

    centres_x, _, _ = get_atom_centres(mode.system).T
    centres_x = centres_x.reshape(*mode.system.n_repeats)[:, *idx]
    centres_x = np.append(centres_x, centres_x[-1] + mode.system.primitive_cell[0, 0])

    (line,) = ax.plot(centres_x, displacement_x)

    ax.set_xlabel("x")
    ax.set_ylabel("x displacement")
    ax.set_xlim(0, centres_x[-1])
    return fig, ax, line


def _get_default_times(mode: NormalMode) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the default times to animate a mode."""
    if mode.omega == 0:
        return np.linspace(0, 1, 20)
    period = 2 * np.pi / mode.omega
    return np.linspace(0, period, 20)


def animate_mode_1d_x(
    mode: NormalMode,
    times: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, ArtistAnimation]:
    fig, ax = get_axis(ax)

    times = times if times is not None else _get_default_times(mode)
    artists: list[list[Line2D]] = []
    for time in times:
        line = plot_mode_1d_x(mode, time=time, idx=idx, ax=ax)[2]
        line.set_color("C0")
        artists.append([line])
    return fig, ax, ArtistAnimation(fig, artists)


def plot_mode_2d_xy(
    mode: NormalMode,
    time: float = 0,
    idx: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Quiver]:
    fig, ax = get_axis(ax)

    displacement = get_mode_displacement(mode, time=time)
    centres_x, centres_y, _ = get_atom_centres(mode.system).T

    displacement = displacement.reshape(*mode.system.n_repeats, 3)
    displacement_xy = displacement[:, :, idx, :2]

    quiver = ax.quiver(
        centres_x.reshape(*mode.system.n_repeats)[:, :, idx],
        centres_y.reshape(*mode.system.n_repeats)[:, :, idx],
        displacement_xy[:, :, 0],
        displacement_xy[:, :, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax, quiver


def animate_mode_2d_xy(
    mode: NormalMode,
    times: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    idx: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, ArtistAnimation]:
    fig, ax = get_axis(ax)

    times = times if times is not None else _get_default_times(mode)
    artists = [[plot_mode_2d_xy(mode, time=t, idx=idx, ax=ax)[2]] for t in times]
    return fig, ax, ArtistAnimation(fig, artists)
