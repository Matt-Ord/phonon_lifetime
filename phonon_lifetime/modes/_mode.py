from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from phonon_lifetime import System


class NormalModes(ABC):
    """Represents all normal modes of a system."""

    # TODO: implement this  # noqa: FIX002
    # @property
    # @abstractmethod
    # def omega(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    #     """A np.array of frequencies for each mode."""

    # @property
    # @abstractmethod
    # def vectors(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    #     """The vector of the mode, an (n_modes, n_atoms, 3) array."""  # noqa: ERA001

    @property
    @abstractmethod
    def n_q(self) -> int:
        """The number of q points in the calculation."""

    @property
    @abstractmethod
    def n_branch(self) -> int:
        """The number of branches in the calculation."""

    @property
    def n_modes(self) -> int:
        """The number of modes in the calculation."""
        return self.n_q * self.n_branch

    @property
    @abstractmethod
    def system(self) -> System:
        """The system that this normal mode belongs to."""

    @abstractmethod
    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> NormalMode:
        """Select the normal mode for a given branch and q point."""


class NormalMode(ABC):
    """Represents a normal mode of the system."""

    @property
    @abstractmethod
    def omega(self) -> float:
        """The frequency of the mode."""

    @property
    @abstractmethod
    def vector(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """The vector of the mode, an (n_atoms, 3) array."""

    @property
    @abstractmethod
    def system(self) -> System:
        """The system that this normal mode belongs to."""
