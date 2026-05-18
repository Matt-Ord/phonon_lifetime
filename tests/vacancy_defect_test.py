import numpy as np

from phonon_lifetime import cell, defect, system


def _vacancy_columns(
    *, n_primitive: int, n_cells: int, defects: list[int]
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    return np.array(
        [
            cell_idx * n_primitive + defect_idx
            for cell_idx in range(n_cells)
            for defect_idx in defects
        ],
        dtype=np.int64,
    )


def test_vacancy_get_pristine_strain_fills_removed_entries_with_zero() -> None:
    pristine = system.build.with_nearest_neighbor_forces(
        cell.build.graphene(mass=10, distance=1.0),
        spring_constant=1.0,
        periodic=(True, False, False),
        threshold=(0.0, 1.1),
    )
    defects = [1]

    defective = defect.with_vacancy_defect(
        pristine,
        defect.VacancyDefect(defects=defects),
    )
    reconstructed = defective.cell.get_pristine_strain(defective)

    n_primitive = pristine.cell.n_atoms
    n_cells = int(np.prod(pristine.strain_repeats))
    vacancy_cols = _vacancy_columns(
        n_primitive=n_primitive,
        n_cells=n_cells,
        defects=defects,
    )
    non_defects = np.array([0], dtype=np.int64)
    non_defect_cols = np.array(
        [cell_idx * n_primitive for cell_idx in range(n_cells)],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(reconstructed.strain.shape, pristine.strain.shape)
    np.testing.assert_array_equal(reconstructed.strain[defects], 0)
    np.testing.assert_array_equal(reconstructed.strain[:, vacancy_cols], 0)
    np.testing.assert_array_equal(
        reconstructed.strain[non_defects[:, None], non_defect_cols[None, :], :, :],
        pristine.strain[non_defects[:, None], non_defect_cols[None, :], :, :],
    )


def test_vacancy_pristine_roundtrip_preserves_defective_tensor() -> None:
    primitive = system.build.with_nearest_neighbor_forces(
        cell.build.graphene(mass=10, distance=1.0),
        spring_constant=1.0,
        periodic=(True, False, False),
        threshold=(0.0, 1.1),
    )
    pristine = system.as_supercell(primitive, n_repeats=(2, 1, 1))

    vacancy = defect.VacancyDefect(defects=[1, 3])
    defective = defect.with_vacancy_defect(pristine, vacancy)

    reconstructed_pristine = defective.cell.get_pristine_strain(defective)
    defective_again = defect.with_vacancy_defect(reconstructed_pristine, vacancy)

    np.testing.assert_array_equal(defective_again.strain, defective.strain)
