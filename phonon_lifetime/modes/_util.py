from typing import TYPE_CHECKING, Any

import numpy as np

from phonon_lifetime.modes import CanonicalMode, NormalModes
from phonon_lifetime.modes._mode import CanonicalModes
from phonon_lifetime.system._repeat import RepeatSystem
from phonon_lifetime.system._system import System

if TYPE_CHECKING:
    from phonon_lifetime.modes._mode import NormalMode


def get_mode_displacement(mode: NormalMode, time: float = 0.0) -> np.ndarray[Any, Any]:
    """Get the displacement of the mode at a given time.

    returns an array of displacements (n_atoms, 3) at the given time.

    """
    out = np.real(mode.vector * np.exp(-1j * mode.omega * time))

    pristine_mass = np.average(mode.system.masses)
    prefactor = np.sqrt(pristine_mass) / np.sqrt(mode.system.masses[:, None])
    return out * (prefactor * np.sqrt(mode.system.n_atoms))


def repeat_mode[S: System](
    mode: NormalMode[S], n_repeats: tuple[int, int, int]
) -> NormalMode[RepeatSystem]:
    """Repeat a mode to create a new mode for a larger system."""
    n_primitive_atoms = mode.system.n_primitive_atoms
    vector = mode.vector.reshape(n_primitive_atoms, *mode.system.n_repeats, 3)
    vector = vector.copy()
    vector /= np.prod(n_repeats) ** 0.5  # Normalize the mode
    new_vector = np.tile(vector, (1, *n_repeats, 1))
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
    vectors = modes.vectors.reshape(modes.n_modes, *modes.system.n_repeats, 3).copy()
    vectors /= np.prod(n_repeats) ** 0.5  # Normalize the modes
    new_vectors = np.tile(vectors, (1, *n_repeats, 1))
    new_vectors = new_vectors.reshape(modes.n_modes, -1)
    return CanonicalModes(
        system=RepeatSystem(n_repeats=n_repeats, system=modes.system),
        omega=modes.omega,
        vectors=new_vectors,
    )
