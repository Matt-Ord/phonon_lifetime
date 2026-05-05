import warnings

import numpy as np
import pytest
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime import StrainSystem, cell, system
from phonon_lifetime.phonon._mesh import (  # noqa: PLC2701
    _q_from_iq,
    get_mesh_phonons,
)


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
