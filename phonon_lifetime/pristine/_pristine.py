import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Any, override

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime import System
from phonon_lifetime.modes import NormalMode, NormalModes
from phonon_lifetime.system._util import get_pristine_force_matrix


@dataclass(kw_only=True, frozen=True)
class PristineMode(NormalMode):
    """A normal mode of a pristine system."""

    _system: PristineSystem
    _omega: float
    """Frequency of one mode (rad/s)."""
    _primitive_vector: np.ndarray
    """Eigenvector of that mode."""
    _q_val: np.ndarray
    """Wave vector for this mode."""

    @property
    @override
    def system(self) -> PristineSystem:
        return self._system

    @property
    @override
    def omega(self) -> float:
        return self._omega

    @property
    def primitive_vector(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """The primitive vector of the mode."""
        return self._primitive_vector

    @override
    @cached_property
    def vector(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        system = self.system
        nx, ny, nz = system.n_repeats
        qx, qy, qz = self._q_val

        # phase(i,j,k) = exp(2πi (qx*i/Nx + qy*j/Ny + qz*k/Nz) - i ω t)
        phx = np.exp(2j * np.pi * qx * (np.arange(nx)))  # (Nx,)
        phy = np.exp(2j * np.pi * qy * (np.arange(ny)))  # (Ny,)
        phz = np.exp(2j * np.pi * qz * (np.arange(nz)))  # (Nz,)
        # the full phase of each atom, shape (Nx, Ny, Nz)
        phase = phx[:, None, None] * phy[None, :, None] * phz[None, None, :]
        phase = np.ravel(phase)
        return phase[..., None] * self.primitive_vector


@dataclass(kw_only=True, frozen=True)
class PristineModes(NormalModes):
    """Result of a normal mode calculation for a phonon system."""

    _system: PristineSystem
    _omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_q, n_branch)
    _modes: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_q, n_atoms*3, n_branch)
    _q_vals: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_q, 3)

    @property
    @override
    def system(self) -> PristineSystem:
        return self._system

    @property
    def q_vals(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """The q values for each mode, shape (n_q, 3)."""
        # TODO: calculate this on the fly # noqa: FIX002
        return self._q_vals

    @property
    def omega(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """The frequencies for each mode, shape (n_q, n_branch)."""
        # TODO: we should provede a better api to make it clear that  # noqa: FIX002
        # we are selecting ie a specific branch
        return self._omega

    @property
    @override
    def n_q(self) -> int:
        return self._omega.shape[0]

    @property
    @override
    def n_branch(self) -> int:
        return self._omega.shape[1]

    @override
    def get_mode(self, branch: int, q: int | tuple[int, int, int]) -> PristineMode:

        iq = q if isinstance(q, int) else np.ravel_multi_index(q, self.system.n_repeats)

        return PristineMode(
            _system=self._system,
            _omega=self._omega[iq, branch],
            _primitive_vector=self._modes[iq, :, branch],
            _q_val=self._q_vals[iq, :],
        )


class PristineSystem(System):
    """Represents a System of atoms."""

    def __init__(
        self,
        *,
        mass: float,
        primitive_cell: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        n_repeats: tuple[int, int, int],
        spring_constant: tuple[float, float, float],
    ) -> None:
        self._mass = mass
        self._primitive_cell = primitive_cell
        self._n_repeats = n_repeats
        self._spring_constant = spring_constant

        assert self.primitive_cell.shape == (3, 3), (
            "Primitive cell should be a 3x3 array of lattice vectors."
        )
        if any(r % 2 == 0 for r in self.n_repeats):
            warnings.warn(
                "Even n_repeats will result in modes which are not periodic.",
                stacklevel=2,
            )

    @property
    @override
    def primitive_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._primitive_cell

    @property
    @override
    def spring_constant(self) -> tuple[float, float, float]:
        return self._spring_constant

    @property
    @override
    def n_repeats(self) -> tuple[int, int, int]:
        return self._n_repeats

    @property
    def mass(self) -> float:
        """Mass of each atom in the system."""
        return self._mass

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.full(self.n_atoms, self.mass)

    @override
    def get_modes(self) -> PristineModes:
        cell = PhonopyAtoms(
            symbols=["C"],
            masses=[self.mass],
            cell=self.primitive_cell,
            scaled_positions=[[0.0, 0.0, 0.0]],
        )

        phonon = Phonopy(
            unitcell=cell,
            supercell_matrix=np.diag(self.n_repeats),
        )

        phonon.force_constants = get_pristine_force_matrix(self)

        phonon.run_mesh(self.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)

        mesh_dict = phonon.get_mesh_dict()
        return PristineModes(
            _system=self,
            _omega=mesh_dict["frequencies"] * 1e12 * 2 * np.pi,
            _modes=mesh_dict["eigenvectors"],
            _q_vals=mesh_dict["qpoints"],  # cspell: disable-line
        )

    @override
    def as_pristine(self) -> PristineSystem:
        return self
