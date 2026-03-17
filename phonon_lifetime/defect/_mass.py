from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from phonon_lifetime.defect._defect import DefectCell, UnitCell

if TYPE_CHECKING:
    import numpy as np

    from phonon_lifetime import StrainSystem


@dataclass(kw_only=True, frozen=True)
class MassDefect:
    """A mass defect in the system."""

    defects: list[tuple[str | None, float, int]]


class MassDefectCell[C: UnitCell = UnitCell](DefectCell[C]):
    """A cell with a mass defect."""

    def __init__(
        self,
        pristine: C,
        defects: MassDefect,
    ) -> None:
        super().__init__(pristine=pristine)
        self._defects = defects

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        masses = self._pristine.masses
        for _symbol, mass, index in self._defects.defects:
            masses[index] = mass
        return masses

    @property
    @override
    def symbols(self) -> list[str]:
        symbols = self._pristine.symbols
        for symbol, _mass, index in self._defects.defects:
            if symbol is not None:
                symbols[index] = symbol
        return symbols

    @property
    @override
    def atom_fractions(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._pristine.atom_fractions

    @override
    def _get_defective_strain_tensor(
        self, strain: StrainSystem[C]
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
        return strain.strain

    @override
    def _get_pristine_strain_tensor(
        self, strain: StrainSystem[C]
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
        return strain.strain

    @override
    def _get_pristine_phonon_vectors(
        self,
        phonon_vectors: np.ndarray[
            tuple[int, int, Literal[3]], np.dtype[np.complex128]
        ],
    ) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]]:
        return phonon_vectors


def with_mass_defect[C: UnitCell](
    pristine: StrainSystem[C],
    defects: MassDefect,
) -> StrainSystem[MassDefectCell[C]]:
    """Create a mass defect cell from a pristine cell."""
    return MassDefectCell(pristine=pristine.cell, defects=defects).get_defect_strain(
        pristine
    )
