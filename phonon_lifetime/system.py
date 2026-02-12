from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(kw_only=True, frozen=True)
class System:
    """Represents a System of atoms."""

    element: str
    primitive_cell: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    spring_constant: tuple[float, float, float]
    n_repeats: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self):  # noqa: ANN204
        assert self.primitive_cell.shape == (3, 3), (
            "Primitive cell should be a 3x3 array of lattice vectors."
        )

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the system."""
        return np.prod(self.n_repeats).item()

    @property
    def supercell_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Get the supercell lattice vectors.

        supercell_cell[i] is the vector (x, y, z) for the i'th lattice vector of the supercell.

        """
        return np.einsum("i,ij->ij", self.n_repeats, self.primitive_cell)


def get_scaled_positions(
    system: System,
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
    """Get the scaled positions of the atoms in the system.

    This gives the fraction along each of the ith lattice vector of the supercell.
    """
    # The index for all sites
    g = np.indices(system.n_repeats).reshape(3, -1)
    return (g / np.array(system.n_repeats)[:, None]).T


def get_atom_centres(
    system: System,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the centres of the atoms in the system."""
    # get_scaled_positions returns the positions along each axis j
    # for each atom i.
    # system.supercell_cell[j] is the cartesian vector for the j'th lattice vector of the supercell.
    return np.einsum("ij,jk->ik", get_scaled_positions(system), system.supercell_cell)
