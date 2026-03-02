from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from phonon_lifetime import System


def get_supercell_cell(
    system: System,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the supercell lattice vectors.

    supercell_cell[i] is the vector (x, y, z) for the i'th lattice vector of the supercell.

    """
    return np.einsum("i,ij->ij", system.n_repeats, system.primitive_cell)


def get_atom_fractions(
    system: System,
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
    """Get the scaled positions of the atoms in the system.

    This gives the fraction along each of the ith lattice vector of the supercell.
    """
    # The index for all sites
    a = system.primitive_atom_fractions.reshape(-1, 1, 3)
    g = np.indices(system.n_repeats).reshape(3, -1).T.reshape(1, -1, 3)
    return (a + g).reshape(-1, 3)  # ty:ignore[invalid-return-type]


def get_atom_supercell_fractions(
    system: System,
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
    """Get the positions of the atoms in the system in cartesian coordinates."""
    return get_atom_fractions(system) / np.array(system.n_repeats)[np.newaxis, :]


def get_atom_centres(
    system: System,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the centres of the atoms in the system."""
    # get_atom_positions returns the positions along each axis j
    # for each atom i.
    # system.supercell_cell[j] is the cartesian vector for the j'th lattice vector of the supercell.
    return np.einsum(
        "ij,jk->ik", get_atom_supercell_fractions(system), get_supercell_cell(system)
    )
