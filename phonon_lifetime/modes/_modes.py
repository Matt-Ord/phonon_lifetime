from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, override

import numpy as np

if TYPE_CHECKING:
    from phonon_lifetime.system import System


class NormalMode(ABC):
    """Represents a normal mode of the system."""

    @property
    @abstractmethod
    def omega(self) -> float:
        """The frequency of the mode."""

    @property
    @abstractmethod
    def vector(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """The vector of the mode."""

    @property
    @abstractmethod
    def system(self) -> System:
        """The system that this normal mode belongs to."""


def get_mode_displacement(mode: NormalMode, time: float = 0.0) -> np.ndarray[Any, Any]:
    """Get the displacement of the mode at a given time.

    returns an array of displacements (n_atoms, 3) at the given time.

    """
    return np.real(mode.vector * np.exp(-1j * mode.omega * time))


class NormalModeResult(ABC):
    """Result of a normal mode calculation for a phonon system."""

    @property
    @abstractmethod
    def n_q(self) -> int:
        """The number of q points in the calculation."""

    @property
    @abstractmethod
    def n_branch(self) -> int:
        """The number of branches in the calculation."""

    @abstractmethod
    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> NormalMode:
        """Get the normal mode for a given branch and q point."""

    def get_all_modes(self) -> list[NormalMode]:
        """Get all the normal modes in the calculation."""
        return [
            self.get_mode(branch, iq)
            for iq in range(self.n_q)
            for branch in range(self.n_branch)
        ]


@dataclass(kw_only=True, frozen=True)
class PristineMode(NormalMode):
    """A normal mode of a pristine system."""

    _system: System
    _omega: float
    """Frequency of one mode (rad/s)."""
    _primitive_vector: np.ndarray
    """Eigenvector of that mode."""
    _q_val: np.ndarray
    """Wave vector for this mode."""

    @property
    @override
    def system(self) -> System:
        return self._system

    @property
    @override
    def omega(self) -> float:
        return self._omega

    @property
    def primitive_vector(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """The primitive vector of the mode."""
        return self._primitive_vector

    @override
    @cached_property
    def vector(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        system = self.system
        nx, ny, nz = system.n_repeats
        qx, qy, qz = self._q_val

        # phase(i,j,k) = exp(2πi (qx*i/Nx + qy*j/Ny + qz*k/Nz) - i ω t)
        phx = np.exp(2j * np.pi * qx * (np.arange(nx)))  # (Nx,)
        phy = np.exp(2j * np.pi * qy * (np.arange(ny)))  # (Ny,)
        phz = np.exp(2j * np.pi * qz * (np.arange(nz)))  # (Nz,)
        # the full phase of each atom, shape (Nx, Ny, Nz)
        phase = phx[:, None, None] * phy[None, :, None] * phz[None, None, :]
        phase = np.ravel(phase)
        return phase[..., None] * self.primitive_vector


@dataclass(kw_only=True, frozen=True)
class PristineNormalModeResult(NormalModeResult):
    """Result of a normal mode calculation for a phonon system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, n_branch)
    modes: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, n_atoms*3, n_branch)
    q_vals: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, dim)

    @property
    @override
    def n_q(self) -> int:
        return self.omega.shape[0]

    @property
    @override
    def n_branch(self) -> int:
        return self.omega.shape[1]

    @override
    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> PristineMode:

        iq = q if isinstance(q, int) else np.ravel_multi_index(q, self.system.n_repeats)

        return PristineMode(
            _system=self.system,
            _omega=self.omega[iq, branch],
            _primitive_vector=self.modes[iq, :, branch],
            _q_val=self.q_vals[iq, :],
        )


@dataclass(kw_only=True, frozen=True)
class VacancyMode(NormalMode):
    """A normal mode of a system with a vacancy."""

    _system: System
    _omega: float
    """Frequency of one mode (rad/s)."""
    _modes: np.ndarray
    """Wave vector for this mode."""
    _vacancy: list[int]

    @property
    @override
    def system(self) -> System:
        return self._system

    @property
    @override
    def vector(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        defective_vector = self._modes

        out = np.zeros((self.system.n_atoms, 3), dtype=np.complex128)
        indices = np.delete(np.arange(self.system.n_atoms), self._vacancy)
        out[indices] = defective_vector

        return out

    @property
    @override
    def omega(self) -> float:
        return self._omega


@dataclass(kw_only=True, frozen=True)
class VacancyNormalModeResult(NormalModeResult):
    """Result of a normal mode calculation for a vacancy system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_branch)
    modes: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_atoms, 3, n_branch)
    vacancy: list[int]

    @property
    @override
    def n_q(self) -> int:
        return 0

    @property
    @override
    def n_branch(self) -> int:
        return self.omega.shape[0]

    @override
    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> NormalMode:

        return VacancyMode(
            _system=self.system,
            _omega=self.omega[branch],
            _modes=self.modes[:, branch],
            _vacancy=self.vacancy,
        )
