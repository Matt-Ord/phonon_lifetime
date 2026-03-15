from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ase import Atoms
from ase.cell import Cell

if TYPE_CHECKING:
    import numpy as np


class UnitCell(ABC):
    """Represents the unit cell of a system."""

    @property
    @abstractmethod
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Mass of the atoms in the unit cell."""

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Symbols of the atoms in the unit cell."""

    @property
    @abstractmethod
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """The primitive cell vectors.

        primitive_cell[i] is the vector (x, y, z) for the i'th lattice vector of the primitive cell.
        """

    @property
    @abstractmethod
    def atom_fractions(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """The positions of the atoms as a fraction of the primitive cell.

        primitive_atom_positions[i] is the position (x, y, z) of the i'th atom in the primitive cell.
        """

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the cell."""
        return self.masses.size


def as_ase_atoms(cell: UnitCell) -> Atoms:
    """Convert a UnitCell to an ASE Atoms object."""
    return Atoms(
        symbols=cell.symbols,
        masses=cell.masses,
        cell=Cell(cell.vectors),
        scaled_positions=cell.atom_fractions,
    )


def get_atom_positions(
    cell: UnitCell,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the positions of the atoms in the cell."""
    return as_ase_atoms(cell).get_positions()
