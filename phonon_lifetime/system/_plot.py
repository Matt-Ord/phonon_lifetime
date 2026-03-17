from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.neighborlist import neighbor_list

from phonon_lifetime import cell
from phonon_lifetime.cell import as_ase_atoms

if TYPE_CHECKING:
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from phonon_lifetime.system import StrainSystem


def plot_xyz(
    system: StrainSystem,
    displacement: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    | None = None,
    *,
    ax: Axes3D | None = None,
    bond_cutoff: float = np.inf,
    scale_bond_lines: bool = True,
) -> tuple[Figure, Axes3D, tuple[PathCollection, Line3DCollection]]:
    fig, ax, (scatter, line_collection) = cell.plot_xyz(
        system.cell,
        displacement=displacement,
        ax=ax,
        bond_cutoff=bond_cutoff,
    )

    if scale_bond_lines:
        as_ase = as_ase_atoms(system.cell)
        as_ase.set_pbc(False)
        bonds = neighbor_list("ijD", as_ase, cutoff=bond_cutoff)
        unit_vectors = bonds[2] / np.linalg.norm(bonds[2], axis=1, keepdims=True)
        linewidths = np.abs(
            np.einsum(
                "na, nab, nb -> n",
                unit_vectors,
                system.strain[bonds[0], bonds[1]],
                unit_vectors,
            )
        )
        linewidths /= np.max(np.abs(linewidths))

        line_collection.set_linewidth(linewidths)

    return fig, ax, (scatter, line_collection)


def plot_xy(
    system: StrainSystem,
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
