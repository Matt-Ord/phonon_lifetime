from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms  # type: ignore[import]


@dataclass(kw_only=True, frozen=True)
class System:
    """Represents a lattice system used for phonon calculations."""

    cell: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    # n_repeats: tuple[int, int, int]
    spring_constant: tuple[float, float, float]
    symbols: list[str]
    scaled_positions: np.ndarray[tuple[int, int], np.dtype[np.floating]]

    @property
    def masses(self) -> np.ndarray:
        """Masses of atoms in the unit cell in atomic mass units (amu)."""
        cell = PhonopyAtoms(
            symbols=self.symbols,
            cell=self.cell,
            scaled_positions=self.scaled_positions,
        )
        return np.asarray(cell.masses, dtype=float)

    @property
    def mass(self) -> float:
        m = self.masses
        if np.allclose(m, m[0]):
            return float(m[0])
        msg = "System has multiple atomic masses; use `.masses` instead of `.mass`."
        raise ValueError(msg)


@dataclass(kw_only=True, frozen=True)
class NormalMode:
    omega: float
    """Frequency of one mode (rad/s)."""
    modes: np.ndarray
    """Eigenvector of that mode."""
    q_val: np.ndarray
    """Wave vector for this mode."""


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

    def get_modes_at_branch(self, branch: int) -> ModesAtBranch:
        """Get the modes at a particular branch."""
        return ModesAtBranch(
            omega=self.omega[:, branch],  # (Nq,)
            modes=self.modes[:, branch, :],  # The second index should be branch.
            q_vals=self.q_vals,  # (Nq, dim)
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


def _build_force_constant_matrix(system):
    """
    FC builder for:
    - 2D square grid with size 3*3 inside the unit cell.
    """
    nx, ny = 1, 1
    kx, ky = system.spring_constant[0], system.spring_constant[1]

    spos = np.asarray(system.scaled_positions, float)  # (nb,3)
    nb = spos.shape[0]  # atoms per defective unit cell

    Lx, Ly = 3 * nx, 3 * ny  # grid size of whole supercell

    # global index: cell (ix,iy) then basis ib
    def gidx(ix: int, iy: int, ib: int) -> int:
        return (iy * nx + ix) * nb + ib

    site2I: dict[tuple[int, int], int] = {}
    # In the 3*3 unit cell case, ix=iy=1, ib=8
    for ix in range(nx):
        for iy in range(ny):
            for ib in range(nb):
                lx = int(3 * spos[ib, 0])
                ly = int(3 * spos[ib, 1])
                X = 3 * ix + lx
                Y = 3 * iy + ly
                site2I[X, Y] = gidx(ix, iy, ib)
    # Site2I marks which sites have atoms
    N = nx * ny * nb
    fc = np.zeros((N, N, 3, 3), float)

    def add_bond(i: int, j: int, k: float, cart: int) -> None:
        fc[i, i, cart, cart] += k
        fc[j, j, cart, cart] += k
        fc[i, j, cart, cart] -= k
        fc[j, i, cart, cart] -= k

    # add +x and +y bonds
    for (X, Y), I in site2I.items():
        J = site2I.get(((X + 1) % Lx, Y))
        # If the right neighbour has atom, J=1, otherwise J=0
        if J is not None:
            add_bond(I, J, kx, cart=0)

        J = site2I.get((X, (Y + 1) % Ly))
        if J is not None:
            add_bond(I, J, ky, cart=1)

    return fc


def calculate_normal_modes(system) -> NormalModeResult:
    cell = PhonopyAtoms(
        symbols=system.symbols,
        cell=system.cell,
        scaled_positions=system.scaled_positions,
    )

    def shift_qmesh_fft_to_physical(q_vals, omega, modes, mesh):
        mesh = np.asarray(mesh, dtype=int)
        dim = np.count_nonzero(mesh > 1)  # active dimensions

        # reshape everything to mesh grid
        q_grid = q_vals.reshape(*mesh, 3)
        omega_grid = omega.reshape(*mesh, *omega.shape[1:])
        modes_grid = modes.reshape(*mesh, *modes.shape[1:])

        # fftshift along reciprocal directions
        axes = tuple(range(dim))
        q_grid = np.fft.fftshift(q_grid, axes=axes)
        omega_grid = np.fft.fftshift(omega_grid, axes=axes)
        modes_grid = np.fft.fftshift(modes_grid, axes=axes)

        # flatten back
        q_vals_s = q_grid.reshape(-1, 3)
        omega_s = omega_grid.reshape(-1, *omega.shape[1:])
        modes_s = modes_grid.reshape(-1, *modes.shape[1:])

        return q_vals_s, omega_s, modes_s

    # No extra supercell here in defected case (3*3 unite cell)
    supercell_matrix = np.diag([1, 1, 1])
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    # Force constants now correspond to the (defected) unit cell basis
    phonon.force_constants = _build_force_constant_matrix(system)

    # n_repeats is used as the q-mesh density
    phonon.run_mesh([3, 3, 1], with_eigenvectors=True, is_mesh_symmetry=False)

    mesh_dict = phonon.get_mesh_dict()
    omega = mesh_dict["frequencies"] * 1e12 * 2 * np.pi
    modes = mesh_dict["eigenvectors"]
    q_vals = mesh_dict["qpoints"]
    print(modes.shape)
    q_vals, omega, modes = shift_qmesh_fft_to_physical(
        q_vals=q_vals, omega=omega, modes=modes, mesh=(3, 3, 1)
    )

    return NormalModeResult(
        system=system,
        omega=omega,
        modes=modes,
        q_vals=q_vals,
    )
