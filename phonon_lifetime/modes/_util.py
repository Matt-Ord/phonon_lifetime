from typing import TYPE_CHECKING, Any

import numpy as np

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
