from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.animation import ArtistAnimation

from phonon_lifetime._util import get_axis, get_axis_3d
from phonon_lifetime.cell import get_atom_positions
from phonon_lifetime.phonon._phonon import get_displacement
from phonon_lifetime.system._plot import plot_xy as plot_system_xy
from phonon_lifetime.system._plot import plot_xyz as plot_system_xyz

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from phonon_lifetime.phonon import (
        DispersionPath,
        DispersionPathPhonons,
        DispersionSegment,
    )
    from phonon_lifetime.phonon._dispersion import DispersionSegmentPhonons
    from phonon_lifetime.phonon._phonon import Phonon


def plot_phonon_1d_x(
    phonon: Phonon,
    time: float = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = get_axis(ax)

    displacement_x = get_displacement(phonon, time=time)[:, 0]

    centres_x, _, _ = get_atom_positions(phonon.system.cell).T

    (line,) = ax.plot(centres_x, displacement_x)

    ax.set_xlabel("x")
    ax.set_ylabel("x displacement")
    ax.set_xlim(0, centres_x[-1])
    return fig, ax, line


def _get_default_times(phonon: Phonon) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the default times to animate a mode."""
    if phonon.omega == 0:
        return np.linspace(0, 1, 20)
    period = 2 * np.pi / phonon.omega
    return np.linspace(0, period, 20)


def animate_phonon_1d_x(
    phonon: Phonon,
    times: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, ArtistAnimation]:
    fig, ax = get_axis(ax)

    times = times if times is not None else _get_default_times(phonon)
    artists: list[list[Line2D]] = []
    for time in times:
        line = plot_phonon_1d_x(phonon, time=time, ax=ax)[2]
        line.set_color("C0")
        artists.append([line])
    return fig, ax, ArtistAnimation(fig, artists)


def plot_phonon_xyz(
    phonon: Phonon,
    time: float = 0,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
    scale_bond_lines: bool = True,
) -> tuple[Figure, Axes3D, tuple[PathCollection, Artist]]:

    displacement = get_displacement(phonon, time=time)
    return plot_system_xyz(
        phonon.system,
        displacement=displacement,
        ax=ax,
        bond_cutoff=bond_cutoff,
        scale_bond_lines=scale_bond_lines,
    )


def animate_phonon_xyz(
    phonon: Phonon,
    times: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
    scale_bond_lines: bool = True,
) -> tuple[Figure, Axes3D, ArtistAnimation]:
    fig, ax = get_axis_3d(ax)

    times = times if times is not None else _get_default_times(phonon)
    artists = [
        plot_phonon_xyz(
            phonon,
            time=t,
            ax=ax,
            bond_cutoff=bond_cutoff,
            scale_bond_lines=scale_bond_lines,
        )[2]
        for t in times
    ]
    return fig, ax, ArtistAnimation(fig, artists)


def plot_phonon_xy(  # noqa: PLR0913
    phonon: Phonon,
    time: float = 0,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
    scale_bond_lines: bool = True,
    scale_displacement: float = 1.0,
) -> tuple[Figure, Axes3D, tuple[PathCollection, Artist]]:

    displacement = get_displacement(phonon, time=time)
    return plot_system_xy(
        phonon.system,
        displacement=displacement * scale_displacement,
        ax=ax,
        bond_cutoff=bond_cutoff,
        scale_bond_lines=scale_bond_lines,
    )


def animate_phonon_xy(  # noqa: PLR0913
    phonon: Phonon,
    times: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
    scale_bond_lines: bool = True,
    scale_displacement: float = 1.0,
) -> tuple[Figure, Axes3D, ArtistAnimation]:
    fig, ax = get_axis_3d(ax)

    times = times if times is not None else _get_default_times(phonon)
    artists = [
        plot_phonon_xy(
            phonon,
            time=t,
            ax=ax,
            bond_cutoff=bond_cutoff,
            scale_bond_lines=scale_bond_lines,
            scale_displacement=scale_displacement,
        )[2]
        for t in times
    ]
    return fig, ax, ArtistAnimation(fig, artists)


def _get_relative_q_values_segment(
    phonons: DispersionSegment,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Get the q values relative to the first point."""
    q_values = phonons.q_values
    first_q = q_values[0]
    return np.linalg.norm(q_values - first_q, axis=-1)


def _set_segment_ticks(
    ax: Axes, labels: tuple[str, str], q: tuple[float, float]
) -> None:
    """Apply major labels, dynamic minor ticks, and vertical lines for a segment."""
    # 1. Steal Matplotlib's default tick spacing
    default_ticks = ax.get_xticks()  # cspell:disable-line
    target_spacing = (
        default_ticks[1] - default_ticks[0] if len(default_ticks) > 1 else q[1] / 5.0
    )

    # 3. Calculate dynamic minor ticks based on the target spacing
    n_divs = max(3, round((q[1] - q[0]) / target_spacing))
    ticks = np.linspace(q[0], q[1], n_divs + 1)
    ax.set_xticks(ticks, minor=True)  # cspell:disable-line
    tick_labels = ["" for _ in ticks]
    tick_labels[0] = labels[0]
    tick_labels[-1] = labels[1]
    ax.set_xticklabels(tick_labels)  # cspell:disable-line
    for pos in q:
        ax.axvline(pos, color="black", linestyle="-", linewidth=0.5)


def plot_dispersion_segment(
    phonons: DispersionSegmentPhonons,
    branch: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = get_axis(ax)

    segment = phonons.segment
    q_vals = _get_relative_q_values_segment(segment)
    omega = phonons.select_phonon(branch=branch).omega

    (line,) = ax.plot(q_vals, omega)  # cspell: disable-line

    ax.set_xlim(0, q_vals[-1])
    ax.set_ylabel("Frequency (THz)")
    _set_segment_ticks(ax, (segment.start[0], segment.end[0]), (0.0, q_vals[-1]))
    return fig, ax, line


def _get_relative_q_values_path(
    phonons: DispersionPath,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], tuple[float, ...]]:
    """Get the q values relative to the first point."""
    out = []
    offset = 0.0
    points = [0.0]
    for segment in phonons:
        q_points = _get_relative_q_values_segment(segment)
        q_points += offset
        offset = q_points[-1]
        out.append(q_points)
        points.append(offset)
    return np.concatenate(out, axis=0), tuple(points)


def _set_path_ticks(ax: Axes, labels: tuple[str, ...], q: tuple[float, ...]) -> None:
    """Apply major labels, dynamic minor ticks, and vertical lines for a path."""
    # 1. Steal Matplotlib's default tick spacing
    default_ticks = ax.get_xticks()  # cspell:disable-line
    target_spacing = (
        default_ticks[1] - default_ticks[0] if len(default_ticks) > 1 else q[-1] / 5.0
    )

    all_ticks = [q[0]]
    all_labels = [labels[0]]

    # 2. Iterate through each segment defined by the points in q
    for i in range(len(q) - 1):
        q_start, q_end = q[i], q[i + 1]

        # Calculate dynamic minor ticks based on the target spacing
        n_divs = max(3, round((q_end - q_start) / target_spacing))

        # np.linspace gives us the divisions; [1:] drops the start point to avoid overlap
        segment_ticks = np.linspace(q_start, q_end, n_divs + 1)[1:]

        segment_labels = ["" for _ in segment_ticks]
        segment_labels[-1] = labels[i + 1]

        all_ticks.extend(segment_ticks)
        all_labels.extend(segment_labels)

    # 3. Apply the combined ticks and labels to the axis
    ax.set_xticks(all_ticks)  # cspell:disable-line
    ax.set_xticklabels(all_labels)  # cspell:disable-line

    # 4. Add vertical lines exactly at the high-symmetry points
    for pos in q:
        ax.axvline(pos, color="black", linestyle="-", linewidth=0.5)


def plot_dispersion_path(
    phonons: DispersionPathPhonons,
    branch: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = get_axis(ax)

    q_vals, points = _get_relative_q_values_path(phonons.path)
    omega = phonons.select_phonon(branch=branch).omega

    (line,) = ax.plot(q_vals, omega)  # cspell: disable-line
    ax.set_xlim(0, q_vals[-1])
    ax.set_ylabel("Frequency (THz)")
    _set_path_ticks(ax, phonons.path.labels, points)
    return fig, ax, line
