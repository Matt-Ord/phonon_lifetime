from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

import numpy as np

if TYPE_CHECKING:
    from phonon_lifetime import System


class NormalModes[S: System](ABC):
    """Represents all normal modes of a system."""

    @property
    @abstractmethod
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """A np.array of frequencies for each mode."""

    @property
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """The vector of the mode, an (n_modes, n_atoms) array."""
        out = np.zeros(
            (self.n_q, self.n_branch, self.system.n_atoms, 3), dtype=np.complex128
        )
        for iq in range(self.n_q):
            for branch in range(self.n_branch):
                mode = self.get_mode(branch=branch, q=iq)
                out[iq, branch] = mode.vector
        return out.reshape(self.n_modes, -1)

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
    def system(self) -> S:
        """The system that this normal mode belongs to."""

    @abstractmethod
    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> NormalMode[S]:
        """Select the normal mode for a given branch and q point."""


class NormalMode[S: System](ABC):
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
    def system(self) -> S:
        """The system that this normal mode belongs to."""

    def as_canonical(self) -> CanonicalMode[S]:
        """Convert this mode to the canonical form."""
        return CanonicalMode(omega=self.omega, vector=self.vector, system=self.system)


class CanonicalMode[S: System](NormalMode[S]):
    """A normal mode in the canonical form, with vectors provided explicitly."""

    def __init__(
        self,
        *,
        omega: float,
        vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        system: S,
    ) -> None:
        self._omega = omega
        self._vector = vector
        self._system = system

    @property
    @override
    def omega(self) -> float:
        """The frequency of the mode."""
        return self._omega

    @property
    @override
    def vector(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """The vector of the mode, an (n_atoms, 3) array."""
        return self._vector

    @property
    @override
    def system(self) -> S:
        """The system that this normal mode belongs to."""
        return self._system
