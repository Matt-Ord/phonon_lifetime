import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast, overload, override

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime.cell import SuperCell, UnitCell
from phonon_lifetime.phonon._phonon import (
    Phonon,
    Phonons,
    as_supercell_phonon,
    as_supercell_phonons,
    get_phonon,
)
from phonon_lifetime.system import StrainSystem

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_mesh_phonons[S: StrainSystem = StrainSystem](
    system: S,
    n_repeats: tuple[int, int, int] = (1, 1, 1),
) -> MeshPhonons[S]:
    """Get the phonon mesh of the system."""
    cell = PhonopyAtoms(
        symbols=system.cell.symbols,
        masses=system.cell.masses.astype(np.float64),
        cell=system.cell.vectors.astype(np.float64),
        scaled_positions=system.cell.atom_fractions.astype(np.float64),
    )

    supercell_n = system.strain_repeats
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Point group symmetries.*")
        phonon = Phonopy(
            unitcell=cell,
            supercell_matrix=np.diag(supercell_n),
        )

    phonon.force_constants = system.strain.astype(np.float64)
    phonon.run_mesh(
        n_repeats,
        with_eigenvectors=True,
        is_mesh_symmetry=False,
        is_gamma_center=True,
    )

    mesh_dict = phonon.get_mesh_dict()

    return MeshPhonons[S](
        system=system,
        omega=(mesh_dict["frequencies"] * 2 * np.pi).reshape(-1),  # ty:ignore[invalid-key]
        vectors=mesh_dict["eigenvectors"].reshape(-1, system.cell.n_atoms, 3),  # ty:ignore[invalid-argument-type, unresolved-attribute, invalid-key]
        n_repeats=n_repeats,
    )


def get_mesh_phonon[S: StrainSystem = StrainSystem](
    system: S,
    q: int | tuple[int, int, int],
    branch: int = 0,
    n_repeats: tuple[int, int, int] = (1, 1, 1),
) -> MeshPhonon[S]:
    """Get the phonon mesh phonon of the system at a given q point and branch."""
    modes = get_phonon(system, q=_q_from_iq(q, n_repeats), branch=branch)
    return MeshPhonon[S](
        system=system, omega=modes.omega, vector=modes.vector, iq=q, n_repeats=n_repeats
    )


def fft_freq_with_nyquist(n: int) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Get the FFT frequencies for a given number of points, with the Nyquist frequency negated to match phonopy."""
    freq = np.fft.fftfreq(n)  # cspell: disable-line
    if n % 2 == 0:
        freq = freq.copy()
        freq[n // 2] = -freq[n // 2]
    return freq


def _q_from_iq(
    iq: int | tuple[int, int, int], n_repeats: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Convert an index to q-values matching phonopy's q points convention."""
    indices = iq_as_stacked(iq, n_repeats)
    q_grids = [fft_freq_with_nyquist(n) for n in n_repeats]
    return cast(
        "tuple[float, float, float]",
        tuple(q_grids[i][indices[i]] for i in range(3)),
    )


def iq_as_flattened(
    iq: tuple[int, int, int] | int, n_repeats: tuple[int, int, int]
) -> int:
    """Convert a q point in the form of a tuple of integers to a flattened index."""
    return (
        iq
        if isinstance(iq, int)
        else np.ravel_multi_index(iq, n_repeats, order="F").item()
    )


def iq_as_stacked(
    iq: int | tuple[int, int, int], n_repeats: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Convert a q point in the form of a flattened index to a tuple of integers."""
    return (
        iq
        if isinstance(iq, tuple)
        else tuple[int, int, int](
            x.item()
            for x in np.unravel_index(iq, n_repeats, order="F")  # ty:ignore[invalid-argument-type]
        )
    )


class MeshPhonon[S: StrainSystem = StrainSystem](Phonon[S]):
    """Represents a normal mode of a system."""

    def __init__(
        self,
        *,
        system: S,
        omega: float,
        vector: np.ndarray[tuple[int, Literal[3]], np.dtype[np.complex128]],
        iq: int | tuple[int, int, int],
        n_repeats: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        self._iq = iq_as_flattened(iq, n_repeats)
        self._n_repeats = n_repeats
        super().__init__(
            system=system,
            omega=omega,
            vector=vector,
            q=_q_from_iq(self._iq, n_repeats),
        )

    @property
    def iq(self) -> int:
        return self._iq

    @property
    def n_repeats(self) -> tuple[int, int, int]:
        return self._n_repeats


class MeshPhonons[S: StrainSystem = StrainSystem](Phonons[S]):
    """Represents all normal modes of a system."""

    def __init__(
        self,
        *,
        system: S,
        omega: np.ndarray[tuple[int], np.dtype[np.floating]],
        vectors: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]],
        n_repeats: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        self._system = system
        self._omega = omega
        self._vectors = vectors
        self._n_repeats = n_repeats

    @property
    def n_repeats(self) -> tuple[int, int, int]:
        return self._n_repeats

    @override
    def __iter__(self) -> Iterator[MeshPhonon[S]]:
        for i in range(self.n_modes):
            yield self[i]

    @override
    def __getitem__(self, idx: int) -> MeshPhonon[S]:
        """Select the normal mode for a given index."""
        iq = idx // self.n_branch
        return MeshPhonon[S](
            system=self.system,
            omega=self.omega[idx],
            vector=self.vectors[idx],
            iq=iq,
            n_repeats=self.n_repeats,
        )

    @overload
    def get_mode_idx(self, branch: int, iq: int | tuple[int, int, int]) -> int: ...

    @overload
    def get_mode_idx(
        self, branch: int, iq: None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.int64]]: ...
    @override
    def get_mode_idx(
        self, branch: int, iq: int | tuple[int, int, int] | None = None
    ) -> int | np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Get the index of a mode by branch and q point."""
        return super().get_mode_idx(
            branch=branch,
            iq=iq_as_flattened(iq, self.n_repeats) if iq is not None else None,
        )

    @overload
    def select_phonon(
        self, branch: int, iq: int | tuple[int, int, int]
    ) -> MeshPhonon[S]: ...

    @overload
    def select_phonon(self, branch: int, iq: None = None) -> MeshPhonons[S]: ...
    def select_phonon(
        self, branch: int, iq: int | tuple[int, int, int] | None = None
    ) -> MeshPhonon[S] | MeshPhonons[S]:
        """Select a single phonon."""
        if iq is None:
            idx = self.get_mode_idx(branch=branch, iq=None)
            return cast(
                "MeshPhonons[S]",
                MeshPhonons[S](
                    system=self.system,
                    omega=self.omega[idx],
                    vectors=self.vectors[idx],
                    n_repeats=self.n_repeats,
                ),
            )

        idx = self.get_mode_idx(branch=branch, iq=iq)
        return self[idx]

    @cached_property
    @override
    def q_values(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """The q values for each mode. Computed to match phonopy's q points."""
        q_grids = [fft_freq_with_nyquist(n) for n in self.n_repeats]

        # Use meshgrid with reversed order to get correct iteration
        grids = np.meshgrid(q_grids[2], q_grids[1], q_grids[0], indexing="ij")
        # Swap back to [x, y, z] order
        return np.array(grids[::-1]).reshape(3, -1).T

    @property
    @override
    def vectors(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]]:
        return self._vectors

    def at_branch(self, branch: int) -> MeshPhonons[S]:
        """Get all phonons at a given branch."""
        idx = self.get_mode_idx(branch=branch, iq=None)
        return cast(
            "MeshPhonons[S]",
            MeshPhonons[S](
                system=self.system,
                omega=self.omega[idx],
                vectors=self.vectors[idx],
                n_repeats=self.n_repeats,
            ),
        )


class GammaPhonon[S: StrainSystem = StrainSystem](MeshPhonon[S]):
    """Represents a normal mode of a system at the Gamma point."""

    def __init__(
        self,
        *,
        system: S,
        omega: float,
        vector: np.ndarray[tuple[int, Literal[3]], np.dtype[np.complex128]],
    ) -> None:
        super().__init__(
            system=system,
            omega=omega,
            vector=vector,
            iq=0,
            n_repeats=(1, 1, 1),
        )


class GammaPhonons[S: StrainSystem = StrainSystem](MeshPhonons[S]):
    """Represents all normal modes of a system at the Gamma point."""

    def __init__(
        self,
        *,
        system: S,
        omega: np.ndarray[tuple[int], np.dtype[np.floating]],
        vectors: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]],
    ) -> None:
        super().__init__(
            system=system,
            omega=omega,
            vectors=vectors,
            n_repeats=(1, 1, 1),
        )

    @override
    def __getitem__(self, idx: int) -> GammaPhonon[S]:
        return GammaPhonon[S](
            system=self.system,
            omega=self.omega[idx],
            vector=self.vectors[idx],
        )

    @override
    def __iter__(self) -> Iterator[GammaPhonon[S]]:
        for i in range(self.n_modes):
            yield self[i]


def get_gamma_phonons[S: StrainSystem = StrainSystem](system: S) -> GammaPhonons[S]:
    """Get the phonon mesh of the system."""
    phonons = get_mesh_phonons(system, n_repeats=(1, 1, 1))

    return GammaPhonons(system=system, omega=phonons.omega, vectors=phonons.vectors)


def get_gamma_phonon[S: StrainSystem = StrainSystem](
    system: S, *, branch: int
) -> GammaPhonon[S]:
    """Get the phonon mesh phonon of the system at a given q point and branch."""
    phonon = get_mesh_phonons(system, n_repeats=(1, 1, 1)).select_phonon(
        branch=branch, iq=0
    )
    return GammaPhonon[S](system=system, omega=phonon.omega, vector=phonon.vector)


def as_gamma_phonons[C: UnitCell = UnitCell](
    phonons: MeshPhonons[StrainSystem[C]],
) -> GammaPhonons[StrainSystem[SuperCell[C]]]:
    """Convert a phonon mesh of a primitive cell to Gamma phonons of the matching supercell."""
    repeat = as_supercell_phonons(phonons, phonons.n_repeats)
    return GammaPhonons[StrainSystem[SuperCell[C]]](
        system=repeat.system, omega=repeat.omega, vectors=repeat.vectors
    )


def as_gamma_phonon[C: UnitCell = UnitCell](
    phonon: MeshPhonon[StrainSystem[C]],
) -> GammaPhonon[StrainSystem[SuperCell[C]]]:
    """Convert a phonon mesh phonon of a primitive cell to a phonon mesh phonon of the corresponding supercell."""
    repeat = as_supercell_phonon(phonon, phonon.n_repeats)
    return GammaPhonon[StrainSystem[SuperCell[C]]](
        system=repeat.system, omega=repeat.omega, vector=repeat.vector
    )
