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


def with_vacancy_defect[C: UnitCell](
    pristine: StrainSystem[C],
    defects: VacancyDefect,
) -> StrainSystem[VacancyDefectCell[C]]:
    """Create a vacancy defect cell from a pristine cell."""
    return VacancyDefectCell(pristine.cell, defects=defects).get_defect_strain(pristine)
