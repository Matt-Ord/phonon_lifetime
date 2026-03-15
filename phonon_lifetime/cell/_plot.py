from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.neighborlist import neighbor_list
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from phonon_lifetime._util import get_axis_3d
from phonon_lifetime.cell import UnitCell, as_ase_atoms, get_atom_positions

if TYPE_CHECKING:
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_xyz(
    cell: UnitCell,
    displacement: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    | None = None,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
) -> tuple[Figure, Axes3D, tuple[PathCollection, Line3DCollection]]:
    fig, ax = get_axis_3d(ax)

    centres = get_atom_positions(cell)
    if displacement is not None:
        centres += displacement

    scatter = ax.scatter(*centres.T, color="C1", s=20)

    as_ase = as_ase_atoms(cell)
    as_ase.set_pbc(False)
    bonds = neighbor_list("ijD", as_ase, cutoff=bond_cutoff)

    line_collection = Line3DCollection(
        np.stack([centres[bonds[0]], centres[bonds[1]]], axis=1),
        colors="black",
        alpha=0.5,
    )
    ax.add_collection3d(line_collection)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal")  # cspell: disable-line

    return fig, ax, (scatter, line_collection)


def plot_xy(
    cell: UnitCell,
    displacement: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    | None = None,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
) -> tuple[Figure, Axes3D, tuple[PathCollection, Line3DCollection]]:
    fig, ax, (scatter, line_collection) = plot_xyz(
        cell,
        displacement=displacement,
        ax=ax,
        bond_cutoff=bond_cutoff,
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
