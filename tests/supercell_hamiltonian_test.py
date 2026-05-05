from typing import TYPE_CHECKING, Any

import numpy as np

from phonon_lifetime import pristine
from phonon_lifetime.defect import (
    VacancyDefect,
    VacancySystem,
)
from phonon_lifetime.system import build

if TYPE_CHECKING:
    from phonon_lifetime.modes._mode import CanonicalModes


def _get_gamma_hamiltonian(
    phonons: CanonicalModes,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    vectors = phonons.vectors.reshape(phonons.omega.size, -1)
    return np.einsum(
        "ij,i,ik->jk",
        vectors,
        phonons.omega**2,
        vectors.conj(),
    )


def test_defective_hamiltonain() -> None:
    """Test that the defective Hamiltonian is constructed correctly for a simple 1D chain."""
    system = build.cubic(
        mass=10, distance=1.0, n_repeats=(101, 1, 1), structure="simple"
    )

    system = pristine.with_nearest_neighbor_forces(
        system, spring_constant=1.0, periodic=(True, False, False), cutoff=1.1
    )

    pristine_phonons = system.get_modes().as_canonical()

    vacancy_system = VacancySystem(
        pristine=system,
        defect=VacancyDefect(defects=[]),
    )

    supercell_phonons = vacancy_system.get_modes().as_canonical()

    np.testing.assert_allclose(
        _get_gamma_hamiltonian(pristine_phonons),
        _get_gamma_hamiltonian(supercell_phonons),
        rtol=1e-5,
        atol=1e-8,
    )
