from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, override

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime import System
from phonon_lifetime.modes import CanonicalMode, NormalModes
from phonon_lifetime.system import get_atom_supercell_fractions, get_supercell_cell

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
        return np.transpose(self._modes)

    @property
    @override
    def system(self) -> MassDefectSystem:
        return self._system

    @property
    @override
    def n_modes(self) -> int:
        return self._omega.shape[0]

    @override
    def __getitem__(self, idx: int) -> MassDefectMode:
        return MassDefectMode(
            system=self._system,
            omega=self._omega[idx],
            vector=self._modes[:, idx].reshape(-1, 3),
        )


@dataclass(kw_only=True, frozen=True)
class MassDefect:
    """A mass defect in the system."""

    defects: list[tuple[str | None, float, int]]


class MassDefectSystem(System):
    """Represents a system with a mass defect."""

    def __init__(
        self,
        *,
        pristine: PristineSystem,
        defect: MassDefect,
    ) -> None:
        self._pristine: PristineSystem = pristine
        self._defect: MassDefect = defect

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
    def strain_tensor(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        return self._pristine.strain_tensor

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
        for _symbol, mass, index in self.defect.defects:
            masses[index] = mass
        return masses

    @property
    @override
    def symbols(self) -> list[str]:
        symbols = self._pristine.symbols
        for symbol, _mass, index in self.defect.defects:
            if symbol is not None:
                symbols[index] = symbol
        return symbols

    @override
    def get_modes(self) -> MassDefectModes:

        cell = PhonopyAtoms(
            symbols=self.symbols,
            masses=self.masses,
            cell=get_supercell_cell(self),
            scaled_positions=get_atom_supercell_fractions(self),
        )

        phonon = Phonopy(
            unitcell=cell, supercell_matrix=np.eye(3), primitive_matrix=np.eye(3)
        )

        phonon.force_constants = self.strain_tensor

        phonon.run_mesh((1, 1, 1), with_eigenvectors=True, is_mesh_symmetry=False)

        mesh_dict = phonon.get_mesh_dict()

        return MassDefectModes(
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
