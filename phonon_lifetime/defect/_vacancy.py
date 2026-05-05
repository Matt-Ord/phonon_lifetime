from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

import numpy as np

from phonon_lifetime.defect._defect import DefectCell, UnitCell

if TYPE_CHECKING:
    from phonon_lifetime import StrainSystem


@dataclass(kw_only=True, frozen=True)
class VacancyDefect:
    """A vacancy defect in the system."""

    defects: list[int]


class VacancyDefectCell[C: UnitCell = UnitCell](DefectCell[C]):
    """A cell with a vacancy defect."""

    def __init__(
        self,
        pristine: C,
        defects: VacancyDefect,
    ) -> None:
        super().__init__(pristine=pristine)
        self._defects = defects

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.delete(self._pristine.masses, self._defects.defects, axis=0)

    @property
    @override
    def symbols(self) -> list[str]:
        symbols = list(self._pristine.symbols)
        for i in sorted(self._defects.defects, reverse=True):
            del symbols[i]
        return symbols

    @property
    @override
    def atom_fractions(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return np.delete(self._pristine.atom_fractions, self._defects.defects, axis=0)

    @override
    def _get_defective_strain_tensor(
        self, strain: StrainSystem[C]
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
        strain_tensor = strain.strain
        n_primitive = strain_tensor.shape[0]
        n_cells = int(np.prod(strain.strain_repeats))

        # Delete vacancies along axis 0 (the primitive cell)
        defective_strain = np.delete(strain_tensor, self._defects.defects, axis=0)

        # Find the indices of these vacancies across the entire supercell (axis 1)
        # Using the mapping: supercell_index = cell_idx * n_primitive + defect_idx
        supercell_defects = [
            cell_idx * n_primitive + defect_idx
            for cell_idx in range(n_cells)
            for defect_idx in self._defects.defects
        ]

        # Delete the repeating vacancies along axis 1 (the supercell)
        return np.delete(defective_strain, supercell_defects, axis=1)

    @override
    def _get_pristine_strain_tensor(
        self, strain: StrainSystem[C]
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
        defect_strain = strain.strain
        n_primitive = self._pristine.n_atoms
        n_cells = int(np.prod(strain.strain_repeats))

        defect_indices = set(self._defects.defects)
        non_defect_indices = np.array(
            [i for i in range(n_primitive) if i not in defect_indices],
            dtype=np.int64,
        )

        expected_rows = len(non_defect_indices)
        if defect_strain.shape[0] != expected_rows:
            msg = (
                "Input strain tensor shape does not match vacancy-defect cell size: "
                f"expected {expected_rows} rows, got {defect_strain.shape[0]}."
            )
            raise ValueError(msg)

        pristine_strain = np.zeros(
            (n_primitive, n_primitive * n_cells, 3, 3),
            dtype=defect_strain.dtype,
        )

        # Keep ordering consistent with _get_defective_strain_tensor: axis 1 is grouped
        # by repeated cells, each containing only non-defect atoms.
        pristine_axis_1 = (
            np.arange(n_cells, dtype=np.int64)[:, None] * n_primitive
            + non_defect_indices[None, :]
        ).ravel()

        pristine_strain[non_defect_indices[:, None], pristine_axis_1[None, :], :, :] = (
            defect_strain
        )

        return pristine_strain  # ty:ignore[invalid-return-type]

    def _get_pristine_phonon_vectors(
        self,
        phonon_vectors: np.ndarray[
            tuple[int, int, Literal[3]], np.dtype[np.complex128]
        ],
    ) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]]:
        # Convert to shape (n_atoms, 3, n_modes) for easier manipulation
        defective_modes = np.einsum("kij->ijk", phonon_vectors)

        n_modes = phonon_vectors.shape[0]
        out = np.zeros((self._pristine.n_atoms, 3, n_modes), dtype=np.complex128)
        indices = np.delete(np.arange(self._pristine.n_atoms), self._defects.defects)
        out[indices] = defective_modes
        return np.einsum("ijk->kij", out)


def with_vacancy_defect[C: UnitCell](
    pristine: StrainSystem[C],
    defects: VacancyDefect,
) -> StrainSystem[VacancyDefectCell[C]]:
    """Create a vacancy defect cell from a pristine cell."""
    return VacancyDefectCell(pristine.cell, defects=defects).get_defect_strain(pristine)
