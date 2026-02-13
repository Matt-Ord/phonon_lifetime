from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime import System
from phonon_lifetime.modes import CanonicalMode, NormalModes
from phonon_lifetime.system import get_scaled_positions, get_supercell_cell
from phonon_lifetime.system._util import get_pristine_force_matrix

if TYPE_CHECKING:
    from phonon_lifetime.pristine import PristineSystem


class MassDefectMode(CanonicalMode["MassDefectSystem"]):
    """A normal mode of a system with a mass defect."""


@dataclass(kw_only=True, frozen=True)
class MassDefectModes(NormalModes["MassDefectSystem"]):
    """Result of a normal mode calculation for a mass defect system."""

    _system: MassDefectSystem
    _omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_branch)
    _modes: np.ndarray[Any, np.dtype[np.complex128]]  # shape = (n_atoms * 3, n_branch)

    @property
    @override
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """A np.array of frequencies for each mode."""
        return self._omega

    @property
    @override
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        return self._modes.transpose(0, 1)

    @property
    @override
    def system(self) -> MassDefectSystem:
        return self._system

    @property
    @override
    def n_q(self) -> int:
        return 1

    @property
    @override
    def n_branch(self) -> int:
        return self._omega.shape[0]

    @override
    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> MassDefectMode:
        return MassDefectMode(
            system=self._system,
            omega=self._omega[branch],
            vector=self._modes[:, branch].reshape(-1, 3),
        )


@dataclass(kw_only=True, frozen=True)
class MassDefect:
    """A mass defect in the system."""

    defects: list[tuple[float, int]]


class MassDefectSystem(System):
    """Represents a system with a mass defect."""

    def __init__(
        self,
        *,
        pristine: PristineSystem,
        defect: MassDefect,
    ) -> None:
        self._pristine = pristine
        self._defect = defect

    @property
    def defect(self) -> MassDefect:
        """The mass defect in the system."""
        return self._defect

    @property
    @override
    def primitive_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._pristine.primitive_cell

    @property
    @override
    def spring_constant(self) -> tuple[float, float, float]:
        return self._pristine.spring_constant

    @property
    @override
    def n_repeats(self) -> tuple[int, int, int]:
        return self._pristine.n_repeats

    @override
    def as_pristine(self) -> PristineSystem:
        return self._pristine

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        masses = self._pristine.masses
        for mass, index in self.defect.defects:
            masses[index] = mass
        return masses

    @override
    def get_modes(self) -> MassDefectModes:
        all_positions = get_scaled_positions(self)

        cell = PhonopyAtoms(
            symbols=["C"] * self.n_atoms,
            masses=self.masses,
            cell=get_supercell_cell(self),
            scaled_positions=all_positions,
        )

        phonon = Phonopy(
            unitcell=cell, supercell_matrix=np.eye(3), primitive_matrix=np.eye(3)
        )

        pristine_force_constants = get_pristine_force_matrix(self)
        phonon.force_constants = pristine_force_constants

        phonon.run_mesh((1, 1, 1), with_eigenvectors=True, is_mesh_symmetry=False)

        mesh_dict = phonon.get_mesh_dict()

        return MassDefectModes(
            _system=self,
            _omega=mesh_dict["frequencies"][0] * 2 * np.pi,
            _modes=mesh_dict["eigenvectors"][0],
        )
