from abc import ABC, abstractmethod
from typing import Literal, Self, overload, override

import numpy as np

from phonon_lifetime import StrainSystem
from phonon_lifetime.cell import SuperCell, UnitCell
from phonon_lifetime.phonon import (
    GammaPhonon,
    GammaPhonons,
    MeshPhonons,
    Phonon,
    Phonons,
    as_gamma_phonons,
)


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

    @abstractmethod
    def _get_pristine_strain_tensor(
        self, strain: StrainSystem[C]
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
        """Get the pristine strain system corresponding to a defective strain system."""
        ...

    def get_pristine_strain(
        self, strain: StrainSystem[DefectCell[C]]
    ) -> StrainSystem[C]:
        """Create a defect cell from a pristine cell and a strain."""
        assert strain.cell == self, "Strain system must be based on the defect cell."
        return StrainSystem[C](
            cell=self.as_pristine(),
            strain_repeats=strain.strain_repeats,
            strain=self._get_pristine_strain_tensor(strain),
        )

    @abstractmethod
    def _get_pristine_phonon_vectors(
        self,
        phonon_vectors: np.ndarray[
            tuple[int, int, Literal[3]], np.dtype[np.complex128]
        ],
    ) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]]:
        """Get the pristine phonon vectors corresponding to a defective phonon vectors."""
        ...

    def get_pristine_phonon(
        self,
        phonon: Phonon[StrainSystem[DefectCell[C]]],
    ) -> Phonon[StrainSystem[C]]:
        """Create a defect cell from a pristine cell and a strain."""
        return Phonon[StrainSystem[C]](
            system=self.get_pristine_strain(phonon.system),
            omega=phonon.omega,
            vector=self._get_pristine_phonon_vectors(np.array([phonon.vector]))[0],
            q=phonon.q,
        )

    @overload
    def get_pristine_phonons(
        self,
        phonon: MeshPhonons[StrainSystem[DefectCell[C]]],
    ) -> MeshPhonons[StrainSystem[C]]: ...

    @overload
    def get_pristine_phonons(
        self,
        phonon: Phonons[StrainSystem[DefectCell[C]]],
    ) -> Phonons[StrainSystem[C]]: ...

    def get_pristine_phonons(
        self,
        phonon: Phonons[StrainSystem[DefectCell[C]]],
    ) -> Phonons[StrainSystem[C]]:
        """Create a defect cell from a pristine cell and a strain."""
        # TODO: not too sure this works?
        if isinstance(phonon, MeshPhonons):
            return MeshPhonons[StrainSystem[C]](
                system=self.get_pristine_strain(phonon.system),
                omega=phonon.omega,
                vectors=self._get_pristine_phonon_vectors(phonon.vectors),
            )
        return Phonons[StrainSystem[C]](
            system=self.get_pristine_strain(phonon.system),
            omega=phonon.omega,
            vectors=self._get_pristine_phonon_vectors(phonon.vectors),
            q_values=phonon.q_values,
        )


def as_pristine_strain_system[C: UnitCell](
    system: StrainSystem[DefectCell[C]],
) -> StrainSystem[C]:
    return system.cell.get_pristine_strain(system)


def as_pristine_phonon[C: UnitCell](
    phonon: Phonon[StrainSystem[DefectCell[C]]],
) -> Phonon[StrainSystem[C]]:
    return phonon.system.cell.get_pristine_phonon(phonon)


@overload
def as_pristine_phonons[C: UnitCell](
    phonons: MeshPhonons[StrainSystem[DefectCell[C]]],
) -> MeshPhonons[StrainSystem[C]]: ...


@overload
def as_pristine_phonons[C: UnitCell](
    phonons: Phonons[StrainSystem[DefectCell[C]]],
) -> Phonons[StrainSystem[C]]: ...


def as_pristine_phonons[C: UnitCell](
    phonons: Phonons[StrainSystem[DefectCell[C]]],
) -> Phonons[StrainSystem[C]]:
    return phonons.system.cell.get_pristine_phonons(phonons)


def as_pristine_gamma_phonon[C: UnitCell](
    phonon: GammaPhonon[StrainSystem[DefectCell[C]]],
) -> GammaPhonon[StrainSystem[C]]:
    as_pristine = as_pristine_phonon(phonon)
    assert np.allclose(as_pristine.q, [0.0, 0.0, 0.0]), (
        "Input phonon must be a Gamma phonon."
    )
    return GammaPhonon(
        system=as_pristine.system,
        omega=as_pristine.omega,
        vector=as_pristine.vector,
    )


def as_pristine_gamma_phonons[C: UnitCell](
    phonons: MeshPhonons[StrainSystem[DefectCell[C]]],
) -> GammaPhonons[StrainSystem[SuperCell[C]]]:
    as_pristine = as_pristine_phonons(phonons)
    return as_gamma_phonons(as_pristine)
