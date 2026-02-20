from typing import TYPE_CHECKING, Any, override

import numpy as np

from phonon_lifetime.modes import CanonicalMode, NormalModes
from phonon_lifetime.modes._mode import CanonicalModes
from phonon_lifetime.pristine import PristineSystem
from phonon_lifetime.system._system import System

if TYPE_CHECKING:
    from phonon_lifetime.modes._mode import NormalMode


def get_mode_displacement(mode: NormalMode, time: float = 0.0) -> np.ndarray[Any, Any]:
    """Get the displacement of the mode at a given time.

    returns an array of displacements (n_atoms, 3) at the given time.

    """
    out = np.real(mode.vector * np.exp(-1j * mode.omega * time))

    pristine_mass = mode.system.as_pristine().mass
    prefactor = np.sqrt(pristine_mass) / np.sqrt(mode.system.masses[:, None])
    return out * (prefactor * np.sqrt(mode.system.n_atoms))


class RepeatSystem(System):
    """A system that repeats another system."""

    def __init__(self, system: System, n_repeats: tuple[int, int, int]) -> None:
        self._system = system
        self._n_repeats = n_repeats

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.tile(
            self._system.masses.reshape(self._system.n_repeats), self._n_repeats
        ).ravel()

    @property
    @override
    def primitive_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._system.primitive_cell

    @property
    @override
    def spring_constant(self) -> tuple[float, float, float]:
        return self._system.spring_constant

    @property
    @override
    def n_repeats(self) -> tuple[int, int, int]:
        return tuple(
            a * b for a, b in zip(self._system.n_repeats, self._n_repeats, strict=True)
        )

    @override
    def as_pristine(self) -> PristineSystem:
        return PristineSystem(
            mass=self._system.as_pristine().mass,
            primitive_cell=self.primitive_cell,
            n_repeats=self.n_repeats,
            spring_constant=self.spring_constant,
        )

    @override
    def get_modes(self) -> NormalModes[System]:
        return repeat_modes(self._system.get_modes(), n_repeats=self._n_repeats)

    @override
    def get_mode(self, idx: int) -> NormalMode[System]:
        return repeat_mode(self._system.get_mode(idx), n_repeats=self._n_repeats)


def repeat_mode[S: System](
    mode: NormalMode[S], n_repeats: tuple[int, int, int]
) -> NormalMode[RepeatSystem]:
    """Repeat a mode to create a new mode for a larger system."""
    vector = mode.vector.reshape(*mode.system.n_repeats, 3)
    vector /= np.prod(n_repeats) ** 0.5  # Normalize the mode
    new_vector = np.tile(vector, (*n_repeats, 1))
    new_vector = new_vector.reshape(-1, 3)
    return CanonicalMode(
        system=RepeatSystem(n_repeats=n_repeats, system=mode.system),
        omega=mode.omega,
        vector=new_vector,
    )


def repeat_modes[S: System](
    modes: NormalModes[S], n_repeats: tuple[int, int, int]
) -> NormalModes[RepeatSystem]:
    """Repeat a set of modes to create new modes for a larger system."""
    vectors = modes.vectors.reshape(modes.n_modes, *modes.system.n_repeats, 3)
    vectors /= np.prod(n_repeats) ** 0.5  # Normalize the modes
    new_vectors = np.tile(vectors, (1, *n_repeats, 1))
    new_vectors = new_vectors.reshape(modes.n_modes, -1)
    return CanonicalModes(
        system=RepeatSystem(n_repeats=n_repeats, system=modes.system),
        omega=modes.omega,
        vectors=new_vectors,
    )
