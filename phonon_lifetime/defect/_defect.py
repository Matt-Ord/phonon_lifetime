from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Self, override

from phonon_lifetime import StrainSystem
from phonon_lifetime.cell import UnitCell
from phonon_lifetime.phonon import Phonon, Phonons

if TYPE_CHECKING:
    import numpy as np


class DefectCell[C: UnitCell = UnitCell](UnitCell, ABC):
    """Represents the supercell of a system with a defect."""

    def __init__(
        self,
        pristine: C,
    ) -> None:
        self._pristine = pristine

    def as_pristine(self) -> C:
        """Get the pristine cell of the system."""
        return self._pristine

    @property
    @override
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._pristine.vectors

    @abstractmethod
    def _get_defective_strain_tensor(
        self, strain: StrainSystem[C]
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
        """Get the defective strain system corresponding to a pristine strain system."""

    def get_defect_strain(self, strain: StrainSystem[C]) -> StrainSystem[Self]:
        """Create a defect cell from a pristine cell and a strain."""
        assert strain.cell == self._pristine, (
            "Strain system must be based on the pristine cell."
        )
        return StrainSystem[Self](
            cell=self,
            strain_repeats=strain.strain_repeats,
            strain=self._get_defective_strain_tensor(strain),
        )


def as_pristine_strain_system[C: UnitCell](
    system: StrainSystem[DefectCell[C]],
) -> StrainSystem[C]:
    return StrainSystem(
        cell=system.cell.as_pristine(),
        strain_repeats=system.strain_repeats,
        strain=system.strain,  # TODO: add in "empty" strain for the vacancy atom
    )


def as_pristine_phonon[C: UnitCell](
    phonon: Phonon[StrainSystem[DefectCell[C]]],
) -> Phonon[StrainSystem[C]]:
    return Phonon(
        system=as_pristine_strain_system(phonon.system),
        omega=phonon.omega,
        vector=phonon.vector,  # TODO: add in "empty" displacements
        q=phonon.q,
    )


def as_pristine_phonons[C: UnitCell](
    phonons: Phonons[StrainSystem[DefectCell[C]]],
) -> Phonons[StrainSystem[C]]:
    return Phonons(
        system=as_pristine_strain_system(phonons.system),
        omega=phonons.omega,
        vectors=phonons.vectors,  # TODO: add in "empty" displacements
        q_values=phonons.q_values,
    )
