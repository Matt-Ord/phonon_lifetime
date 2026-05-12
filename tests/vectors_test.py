import numpy as np

from phonon_lifetime.cell import SuperCell
from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.defect import MassDefect, with_mass_defect
from phonon_lifetime.phonon import as_gamma_phonons, get_gamma_phonons, get_mesh_phonons
from phonon_lifetime.system import with_nearest_neighbor_forces, with_zero_forces


def test_zero_mass_defect_vectors() -> None:
    cell = build_cell.cubic(mass=10, distance=1, structure="simple")
    cell = SuperCell(cell, n_repeats=(7, 1, 1))

    system = with_zero_forces(cell)
    defect = with_mass_defect(pristine=system, defects=MassDefect(defects=[]))

    modes = get_gamma_phonons(defect)
    vectors = modes.vectors

    for i in range(modes.n_modes):
        np.testing.assert_array_equal(vectors[i].reshape(-1, 3), modes[i].vector)


def test_vectors_orthogonality() -> None:
    cell = build_cell.cubic(mass=10, distance=1, structure="simple")
    cell = SuperCell(cell, n_repeats=(7, 1, 1))

    system = with_nearest_neighbor_forces(cell, spring_constant=1.0)

    modes = get_mesh_phonons(system, n_repeats=(7, 1, 1))
    modes = as_gamma_phonons(modes)

    np.testing.assert_allclose(
        np.einsum(
            "ij,ik->jk",
            np.conj(modes.vectors).reshape(modes.n_modes, -1),
            modes.vectors.reshape(modes.n_modes, -1),
        ),
        np.eye(modes.n_modes),
        atol=1e-8,
    )

    defect = with_mass_defect(
        pristine=modes.system,
        defects=MassDefect(defects=[(None, 11, 0)]),
    )
    defect_phonons = get_gamma_phonons(defect)

    np.testing.assert_allclose(
        np.einsum(
            "ij,ik->jk",
            np.conj(defect_phonons.vectors).reshape(modes.n_modes, -1),
            defect_phonons.vectors.reshape(modes.n_modes, -1),
        ),
        np.eye(modes.n_modes),
        atol=1e-8,
    )
