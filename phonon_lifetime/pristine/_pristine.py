import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, overload, override

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime.modes._mode import NormalMode, NormalModes
from phonon_lifetime.system._system import System


def get_crystal_phases(
    system: PristineSystem, q: tuple[float, float, float]
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """Get the crystal phase of each atom in the system."""
    nx, ny, nz = system.n_repeats
    qx, qy, qz = q
    # phase(i,j,k) = exp(2πi (qx*i/Nx + qy*j/Ny + qz*k/Nz) - i ω t)
    phx = np.exp(2j * np.pi * qx * (np.arange(nx)))  # (Nx,)
    phy = np.exp(2j * np.pi * qy * (np.arange(ny)))  # (Ny,)
    phz = np.exp(2j * np.pi * qz * (np.arange(nz)))  # (Nz,)
    # the full phase of each atom, shape (Nx, Ny, Nz)
    phase = phx[:, None, None] * phy[None, :, None] * phz[None, None, :]
    return np.ravel(phase) / np.sqrt(system.n_atoms)


@dataclass(kw_only=True, frozen=True)
class PristineMode(NormalMode["PristineSystem"]):
    """A normal mode of a pristine system."""

    _system: PristineSystem
    _omega: float
    """Frequency of one mode (rad/s)."""
    _primitive_vector: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """Eigenvector of that mode."""
    _q: tuple[float, float, float]
    """Wave vector for this mode."""

    @property
    def q_val(self) -> tuple[float, float, float]:
        return self._q

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
        phases = get_crystal_phases(self.system, self._q)
        n_primitive_atoms = self.system.n_primitive_atoms
        primitive_vector = self.primitive_vector.reshape(n_primitive_atoms, 3)
        return np.einsum("i,jk->jik", phases, primitive_vector).reshape(-1, 3)


@dataclass(kw_only=True, frozen=True)
class PristineModesAtBranch(NormalModes["PristineSystem"]):
    """Result of a normal mode calculation for a phonon system."""

    _system: PristineSystem
    _omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_q)
    # modes with shape = (n_q, n_atoms * 3)
    _modes: np.ndarray[Any, np.dtype[np.complex128]]
    _q_vals: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_q, 3)

    @property
    @override
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self._omega

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
    def n_q(self) -> int:
        return self._omega.shape[0]

    @property
    @override
    def n_modes(self) -> int:
        return self._omega.size

    def get_mode_idx(self, q: int | tuple[int, int, int]) -> int:
        """Get the index of a mode at a q point."""
        return (
            q
            if isinstance(q, int)
            else np.ravel_multi_index(q, self.system.n_repeats).item()
        )

    def as_full(self) -> PristineModes:
        """Convert to PristineModes by adding back the branch dimension."""
        return PristineModes(
            _system=self._system,
            _omega=self._omega[:, None],
            _modes=self._modes[:, :, None],
            _q_vals=self._q_vals,
        )

    @override
    def __getitem__(self, idx: int) -> PristineMode:
        return PristineMode(
            _system=self._system,
            _omega=self._omega[idx],
            _primitive_vector=self._modes[idx, :],
            _q=tuple(self._q_vals[idx, :]),
        )


@dataclass(kw_only=True, frozen=True)
class PristineModes(NormalModes["PristineSystem"]):
    """Result of a normal mode calculation for a phonon system."""

    _system: PristineSystem
    _omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_q, n_branch)
    # modes with shape = (n_q, n_atoms * 3, n_branch)
    _modes: np.ndarray[Any, np.dtype[np.complex128]]
    _q_vals: np.ndarray[Any, np.dtype[np.floating]]  # shape = (n_q, 3)

    @property
    @override
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self._omega.reshape(-1)

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
    def n_q(self) -> int:
        """Number of q points."""
        return self._omega.shape[0]

    @property
    def n_branch(self) -> int:
        """Number of branches."""
        return self._omega.shape[1]

    @property
    @override
    def n_modes(self) -> int:
        return self.n_q * self.n_branch

    @override
    def __getitem__(self, idx: int) -> PristineMode:
        iq = idx // self.n_branch
        ib = idx % self.n_branch

        return PristineMode(
            _system=self._system,
            _omega=self._omega[iq, ib],
            _primitive_vector=self._modes[iq, :, ib],
            _q=tuple(self._q_vals[iq, :]),
        )

    @overload
    def get_mode_idx(self, branch: int, q: int | tuple[int, int, int]) -> int: ...

    @overload
    def get_mode_idx(
        self, branch: int, q: None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.int64]]: ...

    def get_mode_idx(
        self, branch: int, q: int | tuple[int, int, int] | None = None
    ) -> int | np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Get the index of a mode by branch and q point."""
        if q is None:
            return np.arange(self.n_q) * self.n_branch + branch
        iq = (
            q
            if isinstance(q, int)
            else np.ravel_multi_index(q, self.system.n_repeats).item()
        )
        return iq * self.n_branch + branch

    def select_mode(self, branch: int, q: int | tuple[int, int, int]) -> PristineMode:
        """Select a mode by branch and q point."""
        idx = self.get_mode_idx(branch, q)
        return self[idx]

    def at_branch(self, branch: int) -> PristineModesAtBranch:
        return PristineModesAtBranch(
            _system=self._system,
            _omega=self._omega[:, branch],
            _modes=self._modes[:, :, branch],
            _q_vals=self._q_vals,
        )


def _recover_full_forces(
    forces: np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]],
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
    """Recover the full forces from the pristine forces."""
    n_primitive = forces.shape[0]
    n_atoms = forces.shape[1]
    full_forces = np.zeros((n_atoms, n_atoms, 3, 3), dtype=np.float64)
    for i in range(n_atoms):
        i_in_primitive = i % n_primitive
        c_i = np.unravel_index(i // n_primitive, n_repeats)
        for j in range(n_atoms):
            p_j = j % n_primitive
            c_j = np.unravel_index(j // n_primitive, n_repeats)
            cj_relative = tuple((c_j[d] - c_i[d]) % n_repeats[d] for d in range(3))

            j_relative_to_i = (
                np.ravel_multi_index(cj_relative, n_repeats) * n_primitive + p_j
            )
            full_forces[i, j] = forces[i_in_primitive, j_relative_to_i]
    return full_forces  # ty:ignore[invalid-return-type]


class PristineSystem(System):
    """Represents a System of atoms."""

    _forces: np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]

    def __init__(  # noqa: PLR0913
        self,
        *,
        primitive_masses: np.ndarray[tuple[int], np.dtype[np.floating]],
        primitive_cell: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        primitive_atom_fractions: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        n_repeats: tuple[int, int, int],
        forces: np.ndarray[  # TODO: only store pristine forces? # noqa: FIX002
            tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]
        ]
        | None = None,
        primitive_symbols: list[str] | None = None,
    ) -> None:
        self._primitive_masses = primitive_masses
        self._primitive_symbols: list[str] = (
            ["C"] * primitive_masses.size
            if primitive_symbols is None
            else primitive_symbols
        )
        self._primitive_cell = primitive_cell
        self._primitive_atom_fractions = primitive_atom_fractions
        self._n_repeats = n_repeats
        if forces is None:
            self._forces = np.zeros(  # ty:ignore[invalid-assignment]
                (self.n_primitive_atoms, self.n_atoms, 3, 3),
                dtype=np.float64,
            )
        else:
            self._forces = forces
        self._assert_valid()

    def _assert_valid(self) -> None:
        if self._forces.shape != (self.n_primitive_atoms, self.n_atoms, 3, 3):
            msg = f"Forces should have shape (n_primitive_atoms, n_atoms, 3, 3), but got {self._forces.shape}."
            raise ValueError(msg)

        if self.primitive_cell.shape != (3, 3):
            msg = f"Primitive cell should have shape (3, 3), but got {self.primitive_cell.shape}."
            raise ValueError(msg)

        if len(self.primitive_symbols) != self.n_primitive_atoms:
            msg = f"Number of primitive symbols should match number of primitive atoms, but got {len(self.primitive_symbols)} symbols and {self.n_primitive_atoms} primitive atoms."
            raise ValueError(msg)

        if self.primitive_masses.size != self.n_primitive_atoms:
            msg = f"Number of primitive masses should match number of primitive atoms, but got {self.primitive_masses.size} masses and {self.n_primitive_atoms} primitive atoms."
            raise ValueError(msg)

    @property
    @override
    def primitive_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._primitive_cell

    @property
    def pristine_forces(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        return self._forces

    @property
    @override
    def forces(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        return _recover_full_forces(self._forces, self.n_repeats)

    def with_forces(
        self,
        forces: np.ndarray[
            tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]
        ],
    ) -> PristineSystem:
        """Return a new PristineSystem with the same properties but different forces."""
        return PristineSystem(
            primitive_masses=self.primitive_masses,
            primitive_cell=self.primitive_cell,
            primitive_atom_fractions=self.primitive_atom_fractions,
            n_repeats=self.n_repeats,
            forces=forces,
        )

    @property
    @override
    def n_repeats(self) -> tuple[int, int, int]:
        return self._n_repeats

    @property
    def primitive_masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Mass of the atoms in the unit cell."""
        return self._primitive_masses

    @property
    @override
    def symbols(self) -> list[str]:
        """Symbols of all the atoms in the system."""
        n_supercell = np.prod(self.n_repeats).item()
        return [s for _ in range(n_supercell) for s in self.primitive_symbols]

    @property
    def primitive_symbols(self) -> list[str]:
        """Symbols of the atoms in the unit cell."""
        return self._primitive_symbols

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.tile(self.primitive_masses, np.prod(self.n_repeats).item()).ravel()

    @property
    @override
    def n_primitive_atoms(self) -> int:
        return self.primitive_atom_fractions.shape[0]

    @property
    @override
    def primitive_atom_fractions(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._primitive_atom_fractions

    @override
    def get_modes(self) -> PristineModes:
        cell = PhonopyAtoms(
            symbols=self.primitive_symbols,
            masses=self.primitive_masses,
            cell=self.primitive_cell,
            scaled_positions=self.primitive_atom_fractions,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Point group symmetries.*")
            phonon = Phonopy(
                unitcell=cell,
                supercell_matrix=np.diag(self.n_repeats),
            )

        phonon.force_constants = self.pristine_forces
        phonon.run_mesh(self.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)

        mesh_dict = phonon.get_mesh_dict()

        return PristineModes(
            _system=self,
            _omega=mesh_dict["frequencies"] * 2 * np.pi,
            _modes=mesh_dict["eigenvectors"],
            _q_vals=mesh_dict["qpoints"],  # cspell: disable-line
        )

    @override
    def as_pristine(self) -> PristineSystem:
        return self
