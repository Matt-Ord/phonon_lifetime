from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
    def spring_constant(self) -> tuple[float, float, float]:
        """Spring constant for each direction (kx, ky, kz)."""

    @property
    @abstractmethod
    def n_repeats(self) -> tuple[int, int, int]:
        """Number of repeats of the primitive cell in each direction (nx, ny, nz)."""

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the system."""
        return np.prod(self.n_repeats).item()

    @property
    @abstractmethod
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Mass of each atom in the system."""

    @abstractmethod
    def as_pristine(self) -> PristineSystem:
        """Return a new System with no defects."""

    @abstractmethod
    def get_modes(self) -> NormalModes:
        """Get the normal modes of the system."""

    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> NormalMode:
        """Get the normal mode for a given branch and q point."""
        return self.get_modes().get_mode(branch=branch, q=q)
