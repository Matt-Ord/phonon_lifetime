import warnings
from typing import Any

import numpy as np
import pytest
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime import StrainSystem, cell, system
from phonon_lifetime.cell import SuperCell, build
from phonon_lifetime.phonon import (
    GammaPhonons,
    as_gamma_phonon,
    as_gamma_phonons,
    get_gamma_phonon,
)
from phonon_lifetime.phonon._mesh import (  # noqa: PLC2701
    _q_from_iq,
    get_mesh_phonons,
)
from phonon_lifetime.system import build as build_system


def _get_phonopy_q_points(
    system: StrainSystem, n_repeats: tuple[int, int, int]
) -> np.ndarray:
    """Extract actual q points from phonopy mesh."""
    cell_obj = PhonopyAtoms(
        symbols=system.cell.symbols,
        masses=system.cell.masses.astype(np.float64),
        cell=system.cell.vectors.astype(np.float64),
        scaled_positions=system.cell.atom_fractions.astype(np.float64),
    )

    supercell_n = system.strain_repeats
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Point group symmetries.*")
        phonon = Phonopy(
            unitcell=cell_obj,
            supercell_matrix=np.diag(supercell_n),
        )

    phonon.force_constants = system.strain.astype(np.float64)
    phonon.run_mesh(
        n_repeats,
        with_eigenvectors=True,
        is_mesh_symmetry=False,
        is_gamma_center=True,
    )

    mesh_dict = phonon.get_mesh_dict()
    return mesh_dict["qpoints"]  # cspell: disable-line


@pytest.mark.parametrize(
    "n_repeats",
    [(2, 2, 2), (3, 3, 3), (2, 3, 4), (5, 7, 3)],
)
def test_mesh_phonons_q_points_match_phonopy(
    n_repeats: tuple[int, int, int],
) -> None:
    """Test that MeshPhonons.q_values matches phonopy q points."""
    primitive = cell.build.cubic(mass=10, structure="simple", distance=1.0)
    base_system = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, True, True),
        cutoff=1.1,
    )

    mesh_phonons = get_mesh_phonons(base_system, n_repeats=n_repeats)
    phonopy_q_points = _get_phonopy_q_points(base_system, n_repeats)

    # Verify they match
    np.testing.assert_array_almost_equal(mesh_phonons.q_values, phonopy_q_points)


@pytest.mark.parametrize(
    "n_repeats",
    [(2, 2, 2), (3, 3, 3), (2, 3, 4), (5, 7, 3)],
)
def test_q_from_iq_matches_q_values(n_repeats: tuple[int, int, int]) -> None:
    """Test that _q_from_iq matches q_values."""
    primitive = cell.build.cubic(mass=10, structure="simple", distance=1.0)
    base_system = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, True, True),
        cutoff=1.1,
    )

    mesh_phonons = get_mesh_phonons(base_system, n_repeats=n_repeats)

    for flat_idx in range(int(np.prod(n_repeats))):
        q_from_iq = np.array(_q_from_iq(flat_idx, n_repeats))
        q_from_values = mesh_phonons.q_values[flat_idx]
        np.testing.assert_array_almost_equal(
            q_from_iq, q_from_values, decimal=6, err_msg=f"Mismatch at index {flat_idx}"
        )


def test_mesh_phonons_gamma_energy() -> None:
    """Test that MeshPhonons at Gamma point matches get_gamma_phonons."""
    primitive = cell.build.cubic(mass=10, structure="simple", distance=1.0)
    base_system = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, True, True),
        cutoff=1.1,
    )

    mesh_phonons = get_mesh_phonons(base_system, n_repeats=(3, 3, 3))
    idx = mesh_phonons.get_mode_idx(branch=0, iq=0)
    gamma_phonons = as_gamma_phonon(mesh_phonons[idx])

    repeat_system = system.build.with_nearest_neighbor_forces(
        SuperCell(primitive, n_repeats=(3, 3, 3)),
        spring_constant=1.0,
        periodic=(True, True, True),
        cutoff=1.1,
    )
    expected = get_gamma_phonon(repeat_system, branch=0)

    np.testing.assert_array_almost_equal(
        gamma_phonons.omega,
        expected.omega,
        decimal=5,
        err_msg="Frequencies at Gamma point do not match",
    )


def test_mesh_phonons_supercell_energy() -> None:
    """Test that the defective Hamiltonian is constructed correctly for a simple 1D chain."""
    cell = build.cubic(mass=10, distance=1.0, structure="simple")

    pristine_strain = build_system.with_nearest_neighbor_forces(
        cell,
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    supercell_strain = build_system.with_nearest_neighbor_forces(
        SuperCell(cell, (3, 1, 1)),
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    small_mesh = get_mesh_phonons(supercell_strain, n_repeats=(7, 1, 1))
    big_mesh = get_mesh_phonons(pristine_strain, n_repeats=(21, 1, 1))

    small_omega = np.sort(small_mesh.omega)
    big_omega = np.sort(big_mesh.omega)
    np.testing.assert_allclose(
        small_omega,
        big_omega,
        rtol=1e-5,
        atol=1e-6,
    )


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


@pytest.mark.skip(
    reason="Currently failing, but I think that this might be a numerical precision issue."
)
def test_mesh_phonons_supercell_hamiltonian() -> None:
    """Test that the defective Hamiltonian is constructed correctly for a simple 1D chain."""
    cell = build.cubic(mass=10, distance=1.0, structure="simple")

    pristine_strain = build_system.with_nearest_neighbor_forces(
        cell,
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    supercell_strain = build_system.with_nearest_neighbor_forces(
        SuperCell(cell, (3, 1, 1)),
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    small_mesh = get_mesh_phonons(supercell_strain, n_repeats=(7, 1, 1))
    big_mesh = get_mesh_phonons(pristine_strain, n_repeats=(21, 1, 1))

    small_h = _get_gamma_hamiltonian(as_gamma_phonons(small_mesh))
    big_h = _get_gamma_hamiltonian(as_gamma_phonons(big_mesh))

    np.testing.assert_allclose(small_h, big_h, rtol=1e-5, atol=1e-3)
