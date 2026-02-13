from typing import TYPE_CHECKING

import numpy as np

from phonon_lifetime._util import get_axis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from phonon_lifetime.pristine._pristine import PristineModes


def plot_dispersion_1d(
    result: PristineModes,
    branch: int,
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = get_axis(ax)

    q_vals = result.q_vals.reshape(*result.system.n_repeats, -1)[:, *idx, :]
    qx = q_vals[:, 0]
    energies = result.get_dispersion(branch).reshape(*result.system.n_repeats)[:, *idx]

    (line,) = ax.plot(
        np.fft.fftshift(qx),  # cspell: disable-line
        np.fft.fftshift(energies),  # cspell: disable-line
    )

    return fig, ax, line


def plot_dispersion_2d_xy(
    result: PristineModes,
    branch: int,
    idx: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = get_axis(ax)

    q_vals = result.q_vals.reshape(*result.system.n_repeats, 3)[:, :, idx, :]
    qx = q_vals[:, :, 0]
    qy = q_vals[:, :, 1]
    energies = result.get_dispersion(branch).reshape(*result.system.n_repeats)
    energies = energies[:, :, idx]

    mesh = ax.pcolormesh(  # cspell: disable-line
        np.fft.fftshift(qx),  # cspell: disable-line
        np.fft.fftshift(qy),  # cspell: disable-line
        np.fft.fftshift(energies),  # cspell: disable-line
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax, mesh
