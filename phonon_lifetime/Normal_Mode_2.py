from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms  # type: ignore[import]


@dataclass(kw_only=True, frozen=True)
class System:
    """Represents a lattice system used for phonon calculations."""

    element: str
    cell: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    n_repeats: tuple[int, int, int]
    spring_constant: tuple[float, float, float]

    @property
    def mass(self) -> float:
        """Mass of the element in atomic mass units."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=self.cell,
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


@dataclass(kw_only=True, frozen=True)
class PristineNormalMode:
    omega: float
    """Frequency of one mode (rad/s)."""
    modes: np.ndarray
    """Eigenvector of that mode."""
    q_val: np.ndarray
    """Wave vector for this mode."""

    def get_mode_vector(self) -> np.ndarray[Any, Any]: ...


@dataclass(kw_only=True, frozen=True)
class DefectiveNormalMode:
    omega: float
    """Frequency of one mode (rad/s)."""
    modes: np.ndarray
    """Eigenvector of that mode."""
    q_val: np.ndarray
    """Wave vector for this mode."""

    def get_mode_vector(self) -> np.ndarray[Any, Any]: ...


@dataclass(kw_only=True, frozen=True)
class ModesAtBranch:
    omega: np.ndarray[Any, np.dtype[np.floating]]
    """Frequencies ω(q,s) for fixed branch s."""
    modes: np.ndarray[Any, np.dtype[np.floating]]
    """Eigenvectors e(q,s)."""
    q_vals: np.ndarray[Any, np.dtype[np.floating]]
    """q-points for this branch."""


@dataclass(kw_only=True, frozen=True)
class ModesAtQ:
    omega: np.ndarray[Any, np.dtype[np.floating]]
    """Frequencies ω(q_index, s)."""
    modes: np.ndarray[Any, np.dtype[np.floating]]
    """Eigenvectors e(q_index, s)."""
    q_val: np.ndarray[Any, np.dtype[np.floating]]
    """The single q-point for this slice."""


# 2. The NormalModeResult main class
@dataclass(kw_only=True, frozen=True)
class NormalModeResult:
    """Result of a normal mode calculation for a phonon system."""

    system: Any
    omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, Nbranch)
    modes: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, natom*3,Nbranch)
    q_vals: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, dim)

    def get_modes_at_branch(self, branch: int, k_candidates: int = 3):
        if branch not in {0, 1, 2}:
            msg = "branch must be 0 (x), 1 (y), or 2 (z)"
            raise ValueError(msg)

        omega_all = self.omega  # (Nq, Nbranch)
        modes_all = self.modes  # (Nq, Ndof, Nbranch)
        q_vals = self.q_vals

        Nq, _Nbranch = omega_all.shape
        Ndof = modes_all.shape[1]

        omega_sel = np.empty(Nq)
        modes_sel = np.empty((Nq, Ndof))

        for iq in range(Nq):
            # 1. lowest-frequency candidates
            cand = np.argsort(omega_all[iq])[:k_candidates]

            # 2. polarization score
            best_s = None
            best_score = -1.0
            for s in cand:
                e = modes_all[iq, :, s].reshape(-1, 3)
                score = np.sum(np.abs(e[:, branch]) ** 2)
                if score > best_score:
                    best_score = score
                    best_s = s

            omega_sel[iq] = omega_all[iq, best_s]
            modes_sel[iq] = modes_all[iq, :, best_s]

        return ModesAtBranch(
            omega=omega_sel,
            modes=modes_sel,
            q_vals=q_vals,
        )

    # Extract all modes at fixed q index
    def get_ModesAtQPoint(self, q_index: int) -> ModesAtQ:
        return ModesAtQ(
            omega=self.omega[q_index],  # (Nbranch,)
            modes=self.modes[q_index],  # (Nbranch, natom*3)
            q_val=self.q_vals[q_index],  # (dim,)
        )

    @property
    def q_x(self):
        return self.q_vals[..., 0]

    def to_readable(self) -> str:
        return (
            f"Calculating normal modes for system: {self.system}\n"
            "Normal mode frequencies (omega):\n"
            f"{np.array2string(self.omega, precision=6, separator=', ')}\n"
            "Wave vectors (q):\n"
            f"{np.array2string(self.q_vals, precision=6, separator=', ')}\n"
            "Normal modes (eigenvectors):\n"
            f"{np.array2string(self.modes, precision=6, separator=', ')}\n"
        )


"""
# 3. FORCE CONSTANT BUILDER for 2 atoms case
def _build_force_constant_matrix(system):
    n_x, n_y = system.n_repeats[0], system.n_repeats[1]
    kx, ky = system.spring_constant[0], system.spring_constant[1]

    natom = 2
    ncell = n_x * n_y
    N = ncell * natom

    fc = np.zeros((N, N, 3, 3), dtype=float)

    def cell_index(ix: int, iy: int) -> int:
        return ix * n_y + iy

    def gidx(ix: int, iy: int, b: int) -> int:

        return cell_index(ix, iy) * natom + b

    for ix in range(n_x):
        for iy in range(n_y):
            # A site, b=0
            iA = gidx(ix, iy, 0)

            # x-neighbors of A includes B in same cell, and B in left cell
            jBx0 = gidx(ix, iy, 1)
            jBxL = gidx((ix - 1) % n_x, iy, 1)

            fc[iA, iA, 0, 0] += 2 * kx
            fc[iA, jBx0, 0, 0] -= kx
            fc[iA, jBxL, 0, 0] -= kx

            # y-neighbors of A: A above/below
            jAyP = gidx(ix, (iy + 1) % n_y, 0)
            jAyM = gidx(ix, (iy - 1) % n_y, 0)

            fc[iA, iA, 1, 1] += 2 * ky
            fc[iA, jAyP, 1, 1] -= ky
            fc[iA, jAyM, 1, 1] -= ky

            # B atom, b=1
            iB = gidx(ix, iy, 1)

            # x-neighbors of B: A in same cell, and A in right cell
            jAx0 = gidx(ix, iy, 0)
            jAxR = gidx((ix + 1) % n_x, iy, 0)

            fc[iB, iB, 0, 0] += 2 * kx
            fc[iB, jAx0, 0, 0] -= kx
            fc[iB, jAxR, 0, 0] -= kx

            # y-neighbors of B: B above/below
            jByP = gidx(ix, (iy + 1) % n_y, 1)
            jByM = gidx(ix, (iy - 1) % n_y, 1)

            fc[iB, iB, 1, 1] += 2 * ky
            fc[iB, jByP, 1, 1] -= ky
            fc[iB, jByM, 1, 1] -= ky
    return fc
"""


def _build_force_constant_matrix(system):
    n_x, n_y = system.n_repeats[0], system.n_repeats[1]
    kx, ky = system.spring_constant[0], system.spring_constant[1]
    n = n_x * n_y
    fc = np.zeros((n, n, 3, 3), dtype=float)

    def idx(ix: int, iy: int) -> int:
        return iy * n_x + ix

    for ix in range(n_x):
        for iy in range(n_y):
            i = idx(ix, iy)
            # X neighbors
            jx_p = idx((ix + 1) % n_x, iy)
            jx_m = idx((ix - 1) % n_x, iy)
            fc[i, i, 0, 0] += 2 * kx
            fc[i, jx_p, 0, 0] -= kx
            fc[i, jx_m, 0, 0] -= kx
            # Y neighbors
            jy_p = idx(ix, (iy + 1) % n_y)
            jy_m = idx(ix, (iy - 1) % n_y)
            fc[i, i, 1, 1] += 2 * ky
            fc[i, jy_p, 1, 1] -= ky
            fc[i, jy_m, 1, 1] -= ky
    return fc


# 4. MAIN NORMAL-MODE CALCULATOR
def calculate_normal_modes(system) -> NormalModeResult:
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=system.cell,
        scaled_positions=[[0, 0, 0]],
    )

    supercell_matrix = np.diag(system.n_repeats)
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    phonon.force_constants = _build_force_constant_matrix(system)

    phonon.run_mesh(system.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)

    mesh_dict = phonon.get_mesh_dict()

    return NormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"] * 1e12 * 2 * np.pi,
        modes=mesh_dict["eigenvectors"],
        q_vals=mesh_dict["qpoints"],
    )
