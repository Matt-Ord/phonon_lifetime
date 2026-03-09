from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.neighborlist import neighbor_list
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from phonon_lifetime._util import get_axis_3d
from phonon_lifetime.system._util import as_ase_atoms

if TYPE_CHECKING:
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from phonon_lifetime import System


def plot_xyz(
    system: System,
    displacement: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    | None = None,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
    scale_bond_lines: bool = True,
) -> tuple[Figure, Axes3D, tuple[PathCollection, Line3DCollection]]:
    fig, ax = get_axis_3d(ax)

    as_ase = as_ase_atoms(system.as_pristine())
    if displacement is None:
        centres = as_ase.get_positions()
    else:
        centres = as_ase.get_positions() + displacement

    scatter = ax.scatter(*centres.T, color="C1", s=20)

    as_ase.set_pbc(False)
    bonds = neighbor_list("ijD", as_ase, cutoff=bond_cutoff)

    if scale_bond_lines:
        forces = system.forces

        unit_vectors = bonds[2] / np.linalg.norm(bonds[2], axis=1, keepdims=True)
        linewidths = np.abs(
            np.einsum(
                "na, nab, nb -> n",
                unit_vectors,
                forces[bonds[0], bonds[1]],
                unit_vectors,
            )
        )
        linewidths /= np.max(np.abs(linewidths))
    else:
        linewidths = None
    line_collection = Line3DCollection(
        np.stack([centres[bonds[0]], centres[bonds[1]]], axis=1),
        colors="black",
        alpha=0.5,
        linewidths=linewidths,
    )
    ax.add_collection3d(line_collection)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal")  # cspell: disable-line

    return fig, ax, (scatter, line_collection)


def plot_xy(
    system: System,
    displacement: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    | None = None,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
    scale_bond_lines: bool = True,
) -> tuple[Figure, Axes3D, tuple[PathCollection, Line3DCollection]]:
    fig, ax, (scatter, line_collection) = plot_xyz(
        system,
        displacement=displacement,
        ax=ax,
        bond_cutoff=bond_cutoff,
        scale_bond_lines=scale_bond_lines,
    )
    ax.view_init(elev=90, azim=-90)
    # Hide the z-axis and gridlines to make it look like a 2D plot
    ax.set_zticks([])  # cspell: disable-line  # ty:ignore[call-non-callable]
    ax.set_zlabel("")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # ty:ignore[unresolved-attribute] # cspell: disable-line
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # ty:ignore[unresolved-attribute] # cspell: disable-line
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # cspell: disable-line
    ax.grid(visible=False)
    return fig, ax, (scatter, line_collection)
