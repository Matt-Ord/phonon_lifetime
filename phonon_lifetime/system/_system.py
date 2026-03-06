from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

if TYPE_CHECKING:
    from phonon_lifetime.modes import NormalMode, NormalModes
    from phonon_lifetime.pristine import PristineSystem


class System(ABC):
    """Represents a System of atoms."""

    @property
    @abstractmethod
    def primitive_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Primitive cell lattice vectors.

        primitive_cell[i] is the vector (x, y, z) for the i'th lattice vector of the primitive cell.
        """

    @property
    @abstractmethod
    def forces(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        """Force constant matrix for the system.

        Forces[i, j, alpha, beta] is the force constant between the i'th and j'th atom in the system, for each pair of cartesian directions (alpha, beta).
        """

    @property
    @abstractmethod
    def n_repeats(self) -> tuple[int, int, int]:
        """Number of repeats of the primitive cell in each direction (nx, ny, nz)."""

    @property
    @abstractmethod
    def n_primitive_atoms(self) -> int:
        """Number of atoms in the primitive cell."""

    @property
    @abstractmethod
    def primitive_atom_fractions(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """The positions of the atoms as a fraction of the primitive cell.

        primitive_atom_positions[i] is the position (x, y, z) of the i'th atom in the primitive cell.
        """

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the system."""
        return np.prod(self.n_repeats).item() * self.n_primitive_atoms

    @property
    @abstractmethod
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Mass of every atom in the system."""

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Chemical symbol of every atom in the system."""

    @abstractmethod
    def as_pristine(self) -> PristineSystem:
        """Return a new System with no defects."""

    @abstractmethod
    def get_modes(self) -> NormalModes[Self]:
        """Get the normal modes of the system."""

    def get_mode(self, idx: int) -> NormalMode[Self]:
        """Get the normal mode for a given index."""
        return self.get_modes()[idx]
