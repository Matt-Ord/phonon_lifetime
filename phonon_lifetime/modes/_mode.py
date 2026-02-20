from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from phonon_lifetime import System


class NormalModes[S: System](ABC):
    """Represents all normal modes of a system."""

    @property
    @abstractmethod
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """A np.array of frequencies for each mode."""

    @property
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """The vector of the mode, an (n_modes, n_atoms * 3) array."""
        out = np.zeros((self.n_modes, self.system.n_atoms, 3), dtype=np.complex128)
        for i in range(self.n_modes):
            mode = self[i]
            out[i] = mode.vector
        return out.reshape(self.n_modes, -1)

    @property
    @abstractmethod
    def n_modes(self) -> int:
        """The number of modes in the calculation."""

    @property
    @abstractmethod
    def system(self) -> S:
        """The system that this normal mode belongs to."""

    def __iter__(self) -> Iterator[NormalMode[S]]:
        for i in range(self.n_modes):
            yield self[i]

    @abstractmethod
    def __getitem__(self, idx: int) -> NormalMode[S]:
        """Select the normal mode for a given index."""

    def as_canonical(self) -> CanonicalModes[S]:
        """Convert this mode to the canonical form."""
        return CanonicalModes(
            omega=self.omega, vectors=self.vectors, system=self.system
        )


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


class CanonicalModes[S: System](NormalModes[S]):
    """A collection of normal modes in the canonical form, with vectors provided explicitly."""

    def __init__(
        self,
        *,
        omega: np.ndarray[tuple[int], np.dtype[np.floating]],
        vectors: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        system: S,
    ) -> None:
        self._omega = omega
        self._vectors = vectors
        self._system = system

    @property
    @override
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self._omega

    @property
    @override
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        return self._vectors

    @property
    @override
    def system(self) -> S:
        return self._system

    @property
    @override
    def n_modes(self) -> int:
        return self._omega.shape[0]

    @override
    def __getitem__(self, idx: int) -> CanonicalMode[S]:
        return CanonicalMode(
            system=self._system,
            omega=self._omega[idx],
            vector=self._vectors[idx].reshape(-1, 3),
        )
