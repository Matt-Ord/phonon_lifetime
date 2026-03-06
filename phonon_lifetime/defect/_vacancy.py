from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, override

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime import System
from phonon_lifetime.modes import CanonicalMode, NormalModes
from phonon_lifetime.system import (
    get_atom_supercell_fractions,
    get_supercell_cell,
)

if TYPE_CHECKING:
    from phonon_lifetime.pristine import PristineSystem


class VacancyMode(CanonicalMode["VacancySystem"]):
    """A normal mode of a vacancy defect system."""


@dataclass(kw_only=True, frozen=True)
class VacancyModes(NormalModes["VacancySystem"]):
    """Result of a normal mode calculation for a phonon system."""

    _system: VacancySystem
    _omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_branch)
    # modes with shape = (n_atoms * 3, n_branch)
    _modes: np.ndarray[tuple[int, int], np.dtype[np.complex128]]

    @property
    @override
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """A np.array of frequencies for each mode."""
        return self._omega

    @property
    @override
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        defective_modes = self._modes.reshape(-1, 3, self.n_modes)

        out = np.zeros((self.system.n_atoms, 3, self.n_modes), dtype=np.complex128)
        indices = np.delete(np.arange(self.system.n_atoms), self.system.defect.defects)
        out[indices] = defective_modes
        return out.reshape(self.system.n_atoms * 3, self.n_modes).T

    @property
    @override
    def system(self) -> VacancySystem:
        return self._system

    @property
    @override
    def n_modes(self) -> int:
        return self._omega.shape[0]

    @override
    def __getitem__(self, idx: int) -> VacancyMode:
        defective_vector = self._modes[:, idx].reshape(-1, 3)
        vector = np.zeros((self.system.n_atoms, 3), dtype=np.complex128)
        indices = np.delete(np.arange(self.system.n_atoms), self.system.defect.defects)
        vector[indices] = defective_vector

        return VacancyMode(
            system=self._system,
            omega=self._omega[idx],
            vector=vector,
        )


@dataclass(kw_only=True, frozen=True)
class VacancyDefect:
    """A vacancy defect in the system."""

    defects: list[int]


class VacancySystem(System):
    """Represents a system with a vacancy."""

    def __init__(
        self,
        *,
        pristine: PristineSystem,
        defect: VacancyDefect,
    ) -> None:
        self._pristine = pristine
        self._defect = defect

    @property
    def defect(self) -> VacancyDefect:
        """The vacancy defect in the system."""
        return self._defect

    @property
    @override
    def primitive_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._pristine.primitive_cell

    @property
    @override
    def forces(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        full_forces = self._pristine.forces
        full_forces[self.defect.defects] = 0
        full_forces[:, self.defect.defects] = 0
        return full_forces

    @property
    @override
    def n_repeats(self) -> tuple[int, int, int]:
        return self._pristine.n_repeats

    @override
    def as_pristine(self) -> PristineSystem:
        return self._pristine

    @property
    @override
    def symbols(self) -> list[str]:
        return self._pristine.symbols

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self._pristine.masses

    @override
    def get_modes(self) -> VacancyModes:
        vacancy = self.defect.defects
        all_positions = get_atom_supercell_fractions(self)

        symbols = list(self.symbols)
        for i in sorted(vacancy, reverse=True):
            del symbols[i]

        cell = PhonopyAtoms(
            symbols=symbols,
            masses=np.delete(self.masses, vacancy, axis=0),
            cell=get_supercell_cell(self),
            scaled_positions=np.delete(all_positions, self.defect.defects, axis=0),
        )

        phonon = Phonopy(
            unitcell=cell, supercell_matrix=np.eye(3), primitive_matrix=np.eye(3)
        )

        pristine_force_constants = self.forces
        phonon.force_constants = np.delete(
            np.delete(pristine_force_constants, vacancy, axis=0), vacancy, axis=1
        )

        phonon.run_mesh((1, 1, 1), with_eigenvectors=True, is_mesh_symmetry=False)

        mesh_dict = phonon.get_mesh_dict()

        return VacancyModes(
            _system=self,
            _omega=mesh_dict["frequencies"][0] * 2 * np.pi,
            _modes=mesh_dict["eigenvectors"][0],
        )

    @property
    def n_primitive_atoms(self) -> int:
        return self._pristine.n_primitive_atoms

    @property
    def primitive_atom_fractions(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._pristine.primitive_atom_fractions
