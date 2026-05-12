from typing import Any

import numpy as np

from phonon_lifetime.cell import SuperCell, build
from phonon_lifetime.phonon import (
    GammaPhonons,
    as_gamma_phonons,
    get_gamma_phonons,
    get_mesh_phonons,
)
from phonon_lifetime.system import build as build_system


def _get_gamma_hamiltonian(
    phonons: GammaPhonons,
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
    cell = build.cubic(mass=10, distance=1.0, structure="simple")
    n_repeats = (51, 1, 1)

    pristine_strain = build_system.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, False, False), cutoff=1.1
    )
    pristine_phonons = get_mesh_phonons(pristine_strain, n_repeats=n_repeats)
    pristine_phonons = as_gamma_phonons(pristine_phonons)

    supercell_strain = build_system.with_nearest_neighbor_forces(
        SuperCell(cell, n_repeats),
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )
    supercell_phonons = get_gamma_phonons(supercell_strain)

    np.testing.assert_allclose(
        _get_gamma_hamiltonian(pristine_phonons),
        _get_gamma_hamiltonian(supercell_phonons),
        rtol=1e-5,  # cspell:disable-line
        atol=1e-8,
    )
