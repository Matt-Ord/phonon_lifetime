from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from phonon_lifetime.modes._modes import get_mode_displacement
from phonon_lifetime.system import get_atom_centres

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.lines import Line2D

    from phonon_lifetime import PristineNormalModeResult
    from phonon_lifetime.modes import NormalMode


def _get_axis(ax: Axes | None) -> tuple[Figure, Axes]:
    """Get the axis to plot on."""
    if ax is None:
        fig, ax = plt.subplots()
        assert ax is not None
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax


def plot_mode_2d_xy(
    mode: NormalMode, time: float = 0, idx: int = 0, *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = _get_axis(ax)

    displacement = get_mode_displacement(mode, time=time)
    centres_x, centres_y, _ = get_atom_centres(mode.system).T

    displacement = displacement.reshape(*mode.system.n_repeats, 3)
    displacement_xy = displacement[:, :, idx, :2]

    ax.quiver(
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
    ax.margins(0.5)
    return fig, ax


def plot_mode_1d_x(
    mode: NormalMode,
    time: float = 0,
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = _get_axis(ax)

    displacement = get_mode_displacement(mode, time=time)
    displacement = displacement.reshape(*mode.system.n_repeats, 3)
    displacement_x = displacement[:, *idx, 0]

    centres_x, _, _ = get_atom_centres(mode.system).T
    centres_x = centres_x.reshape(*mode.system.n_repeats)[:, *idx]

    ax.plot(centres_x, displacement_x)

    ax.set_xlabel("x")
    ax.set_ylabel("x displacement")
    ax.set_xlim(0, mode.system.n_repeats[0] * mode.system.primitive_cell[0, 0])
    return fig, ax


def plot_1d_dispersion(
    result: PristineNormalModeResult,
    branch: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = _get_axis(ax)
    qx = result.q_vals[:, 0]
    energies = result.omega[:, branch]

    (line,) = ax.plot(
        np.fft.fftshift(qx),
        np.fft.fftshift(energies),
    )

    return fig, ax, line


def plot_2d_dispersion(
    result: PristineNormalModeResult,
    branch: int,
    idx: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = _get_axis(ax)

    q_vals = result.q_vals.reshape(*result.system.n_repeats, 3)[..., idx, :]
    qx = q_vals[..., 0]
    qy = q_vals[..., 1]
    energies = result.omega[:, branch].reshape(*result.system.n_repeats)[..., idx]

    mesh = ax.pcolormesh(
        np.fft.fftshift(qx),
        np.fft.fftshift(qy),
        np.fft.fftshift(energies),
    )

    return fig, ax, mesh
