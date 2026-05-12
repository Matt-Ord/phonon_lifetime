import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime.cell import SuperCell, UnitCell
from phonon_lifetime.system import StrainSystem
from phonon_lifetime.system._system import as_supercell

if TYPE_CHECKING:
    from collections.abc import Iterator


class Phonon[S: StrainSystem = StrainSystem]:
    """Represents a normal mode of a system."""

    def __init__(
        self,
        *,
        system: S,
        omega: float,
        vector: np.ndarray[tuple[int, Literal[3]], np.dtype[np.complex128]],
        q: tuple[float, float, float],
    ) -> None:
        self._system = system
        self._omega = omega
        self._vector = vector
        self._q = q

    @property
    def system(self) -> S:
        """Get the system of the mode."""
        return self._system

    @property
    def omega(self) -> float:
        """The frequency of the mode."""
        return self._omega

    @property
    def q(self) -> tuple[float, float, float]:
        """The q point of the mode."""
        return self._q

    @cached_property
    def vector(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.complex128]]:
        """Get the vector of the mode, an (n_atoms, 3) array."""
        return self._vector


class Phonons[S: StrainSystem = StrainSystem]:
    """Represents a collection of phonon modes."""

    def __init__(
        self,
        *,
        system: S,
        omega: np.ndarray[tuple[int], np.dtype[np.floating]],
        q_values: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]],
        vectors: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]],
    ) -> None:
        self._system = system
        self._omega = omega
        self._q_values = q_values
        self._vectors = vectors
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.omega.shape[0] != self._vectors.shape[0]:
            msg = f"Number of frequencies should match number of eigenvectors, but got {self._omega.shape[0]} frequencies and {self._vectors.shape[0]} eigenvectors."
            raise ValueError(msg)
        if self.system.cell.n_atoms != self._vectors.shape[1]:
            msg = f"Number of atoms in the system should match the second dimension of the eigenvectors, but got {self.system.cell.n_atoms} atoms and {self._vectors.shape[1]} in the eigenvectors."
            raise ValueError(msg)
        if self._vectors.shape[2] != 3:  # noqa: PLR2004
            msg = f"The last dimension of the eigenvectors should be 3, but got {self._vectors.shape[2]}."
            raise ValueError(msg)

    @property
    def n_q(self) -> int:
        """The number of q points in the calculation."""
        return self.q_values.shape[0]

    @property
    def n_branch(self) -> int:
        """The number of bands in the calculation."""
        return self.n_modes // self.n_q

    @property
    def q_values(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        """The q values for each phonon."""
        return self._q_values

    @property
    def omega(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """A np.array of frequencies for each mode."""
        return self._omega.ravel()

    @property
    def vectors(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]]:
        """The vector of the mode, an (n_modes, n_atoms, 3) array."""
        return self._vectors

    @property
    def n_modes(self) -> int:
        """The number of modes in the calculation."""
        return self._omega.size

    @property
    def system(self) -> S:
        """The system that this normal mode belongs to."""
        return self._system

    def __iter__(self) -> Iterator[Phonon[S]]:
        for i in range(self.n_modes):
            yield self[i]

    @overload
    def get_mode_idx(self, branch: int, iq: int) -> int: ...

    @overload
    def get_mode_idx(
        self, branch: int, iq: None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.int64]]: ...

    def get_mode_idx(
        self, branch: int, iq: int | None = None
    ) -> int | np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Get the index of a mode by branch and q point.

        Modes are indexed by (i_q, i_branch), where i_q is the
        index of the q point and i_branch is the index of the branch.
        """
        if iq is None:
            return np.arange(self.n_q) * self.n_branch + branch

        return iq * self.n_branch + branch

    @overload
    def select_phonon(self, branch: int, iq: int) -> Phonon[S]: ...

    @overload
    def select_phonon(self, branch: int, iq: None = None) -> Phonons[S]: ...

    def select_phonon(
        self, branch: int, iq: int | None = None
    ) -> Phonon[S] | Phonons[S]:
        """Select a single phonon by branch and q point."""
        if iq is None:
            idx = self.get_mode_idx(branch=branch, iq=None)
            return cast(
                "Phonons[S]",
                Phonons(
                    system=self.system,
                    omega=self.omega[idx],
                    vectors=self.vectors[idx],
                    q_values=self.q_values,
                ),
            )

        idx = self.get_mode_idx(branch=branch, iq=iq)
        return self[idx]

    def __getitem__(self, idx: int) -> Phonon[S]:
        """Select the normal mode for a given index."""
        iq = idx // self.n_branch
        return Phonon[S](
            system=self.system,
            omega=self.omega[idx],
            vector=self.vectors[idx],
            q=tuple(self.q_values[iq]),
        )


def force_constants_from_strain(
    system: StrainSystem,
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
    """Convert the strain from the format used in StrainSystem to the format expected by Phonopy."""
    strain = system.strain.astype(np.float64)
    # Phonopy uses a different ordering to standard ASE conventions
    return np.einsum(
        # cspell:disable-next-line  # noqa: ERA001
        "ijklm->ikjlm",
        strain.reshape(
            system.cell.n_atoms,
            np.prod(system.strain_repeats).item(),
            system.cell.n_atoms,
            3,
            3,
        ),
    ).reshape(strain.shape)


def _build_phonopy_system(system: StrainSystem) -> Phonopy:
    primitive_cell = system.cell
    cell = PhonopyAtoms(
        symbols=primitive_cell.symbols,
        masses=primitive_cell.masses.astype(np.float64),
        cell=primitive_cell.vectors.astype(np.float64),
        scaled_positions=primitive_cell.atom_fractions.astype(np.float64),
    )

    supercell_n = system.strain_repeats
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Point group symmetries.*")
        phonopy_system = Phonopy(
            unitcell=cell,
            supercell_matrix=np.diag(supercell_n),
        )

    phonopy_system.force_constants = force_constants_from_strain(system)
    return phonopy_system


def get_phonons[S: StrainSystem = StrainSystem](
    system: S, *, q_values: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
) -> Phonons[S]:
    """Get a set of phonons for the system."""
    phonopy_system = _build_phonopy_system(system)
    # cspell:disable-next-line  # noqa: ERA001
    phonopy_system.run_qpoints(q_values.astype(np.float64), with_eigenvectors=True)
    # cspell:disable-next-line  # noqa: ERA001
    mesh_dict = phonopy_system.get_qpoints_dict()
    # eigenvectors in n_q, (n_atoms x 3), n_bands
    vectors = np.einsum("ijk -> ikj", mesh_dict["eigenvectors"]).reshape(  # ty:ignore[no-matching-overload]
        -1, system.cell.n_atoms, 3
    )
    return Phonons(
        system=system,
        omega=mesh_dict["frequencies"] * 2 * np.pi,
        vectors=vectors,
        q_values=q_values,
    )


def get_phonon[S: StrainSystem = StrainSystem](
    system: S,
    q: tuple[float, float, float],
    branch: int = 0,
) -> Phonon[S]:
    """Get the normal mode of the system at a given q point and branch."""
    phonons = get_phonons(system, q_values=np.array([q]))
    return phonons.select_phonon(branch=branch, iq=0)


def _get_crystal_phases(
    n_repeats: tuple[int, int, int],
    q: tuple[float, float, float],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """Get the crystal phase of each atom in the system."""
    nx, ny, nz = n_repeats
    qx, qy, qz = q
    # q is already expressed in reduced coordinates for the target supercell.
    # phase(i,j,k) = exp(2πi (qx*i + qy*j + qz*k) - i ω t)
    phx = np.exp(2j * np.pi * qx * (np.arange(nx)))  # (Nx,)
    phy = np.exp(2j * np.pi * qy * (np.arange(ny)))  # (Ny,)
    phz = np.exp(2j * np.pi * qz * (np.arange(nz)))  # (Nz,)
    # the full phase of each atom, shape (Nx, Ny, Nz)
    phase = phx[:, None, None] * phy[None, :, None] * phz[None, None, :]
    return np.ravel(phase) / np.sqrt(np.prod(n_repeats))


def _get_supercell_vector(
    phonon: Phonon,
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.complex128]]:
    n_primitive_atoms = phonon.system.cell.n_atoms
    phases = _get_crystal_phases(n_repeats, phonon.q)

    primitive_vector = phonon.vector.reshape(n_primitive_atoms, 3)
    return np.einsum("i,jk->ijk", phases, primitive_vector).reshape(-1, 3)


def as_supercell_phonon[C: UnitCell = UnitCell](
    phonon: Phonon[StrainSystem[C]],
    n_repeats: tuple[int, int, int],
) -> Phonon[StrainSystem[SuperCell[C]]]:
    """Convert a phonon mesh phonon of a primitive cell to a phonon mesh phonon of the corresponding supercell."""
    return Phonon[StrainSystem[SuperCell[C]]](
        system=as_supercell(phonon.system, n_repeats),
        omega=phonon.omega,
        vector=_get_supercell_vector(phonon, n_repeats),
        q=cast(
            "tuple[float, float, float]",
            tuple(qi / n for qi, n in zip(phonon.q, n_repeats, strict=True)),
        ),
    )


def _get_supercell_vectors(
    phonon: Phonons,
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]]:
    out = np.empty(
        (phonon.n_modes, phonon.system.cell.n_atoms * np.prod(n_repeats), 3),
        dtype=np.complex128,
    )
    for i in range(phonon.n_modes):
        out[i] = _get_supercell_vector(phonon[i], n_repeats)
    return out


def as_supercell_phonons[C: UnitCell = UnitCell](
    phonons: Phonons[StrainSystem[C]],
    n_repeats: tuple[int, int, int],
) -> Phonons[StrainSystem[SuperCell[C]]]:
    """Convert a phonon mesh of a primitive cell to a phonon mesh of the corresponding supercell."""
    return Phonons[StrainSystem[SuperCell[C]]](
        system=as_supercell(phonons.system, n_repeats),
        omega=phonons.omega,
        vectors=_get_supercell_vectors(phonons, n_repeats),
        q_values=phonons.q_values / np.array(n_repeats),
    )


def get_displacement(
    phonon: Phonon, time: float = 0.0
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
    """Get the displacement of the mode at a given time.

    returns an array of displacements (n_atoms, 3) at the given time.

    """
    out = np.real(phonon.vector * np.exp(-1j * phonon.omega * time))

    pristine_mass = np.average(phonon.system.cell.masses)
    prefactor = np.sqrt(pristine_mass) / np.sqrt(phonon.system.cell.masses[:, None])
    return out * (prefactor * np.sqrt(phonon.system.cell.n_atoms))
