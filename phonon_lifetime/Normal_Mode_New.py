from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms  # type: ignore[import]


@dataclass(kw_only=True, frozen=True)
class System:
    element: str
    primitive_cell: (
        np.ndarray
    )  # It's not physically primitive cell, just the atom position
    spring_constant: tuple[float, float, float]
    Defected_cell_size: tuple[int, int, int] = (1, 1, 1)
    vacancy: tuple[int, int, int] | None = (
        None  # vacancy can be the coordinate index of the missing atom or none
    )
    n_repeats: tuple[int, int, int] = (1, 1, 1)

    @property
    def unit_cell(self) -> np.ndarray:
        # Lattice vector of unite cell
        p = np.asarray(self.primitive_cell, float)
        if self.vacancy is None:
            return p  # pristine case
        Nx, Ny, Nz = self.Defected_cell_size
        return np.diag([Nx, Ny, Nz]) @ p  # For defected lattice, the

    @property
    def unit_scaled_positions(self) -> np.ndarray:
        if self.vacancy is None:
            return np.array([[0.0, 0.0, 0.0]], float)

        Nx, Ny, Nz = self.Defected_cell_size
        vx, vy, vz = self.vacancy
        g = (
            np.indices((Nx, Ny, Nz)).reshape(3, -1).T
        )  # g is (Nx,Ny,Nz,3), which is coordinates for all sites

        # mask out vacancy site
        mask = ~((g[:, 0] == vx) & (g[:, 1] == vy) & (g[:, 2] == vz))

        g = g[mask].astype(float)

        # convert to scaled positions in the defect unit cell
        g[:, 0] /= Nx
        g[:, 1] /= Ny
        g[:, 2] /= Nz
        return g

    @property
    def unit_symbols(self) -> list[str]:
        return [self.element] * len(self.unit_scaled_positions)

    def get_atom_centres(self) -> np.ndarray[Any, Any]:
        # build occupancy mask

        if self.vacancy is not None:
            nx, ny, nz = (
                self.Defected_cell_size[0],
                self.Defected_cell_size[1],
                self.Defected_cell_size[2],
            )
            occ = np.ones((nx, ny, nz), dtype=bool)
            vx, vy, vz = self.vacancy
            occ[vx % nx, vy % ny, vz % nz] = False
            gx, gy, gz = np.indices((nx, ny, nz))
            gx = gx[occ]
            gy = gy[occ]
            gz = gz[occ]
        else:
            nx, ny, nz = self.n_repeats[0], self.n_repeats[1]
            gx, gy, gz = np.indices((nx, ny, nz))

        # grid indices, vacancy removed
        a1 = self.primitive_cell[0]
        a2 = self.primitive_cell[1]
        R = gx[:, None] * a1[None, :] + gy[:, None] * a2[None, :]

        X = R[:, 0]
        Y = R[:, 1]

        return X, Y


def build_force_constant_matrix(system):
    Nx, Ny = system.Defected_cell_size[0], system.Defected_cell_size[1]
    kx, ky = system.spring_constant[0], system.spring_constant[1]

    def idx(ix: int, iy: int) -> int:
        return iy * Nx + ix

    # 1) pristine fc on full grid
    n = Nx * Ny
    fc = np.zeros((n, n, 3, 3), float)

    for ix in range(Nx):
        for iy in range(Ny):
            i = idx(ix, iy)
            jx_p = idx((ix + 1) % Nx, iy)
            jx_m = idx((ix - 1) % Nx, iy)
            fc[i, i, 0, 0] += 2 * kx
            fc[i, jx_p, 0, 0] -= kx
            fc[i, jx_m, 0, 0] -= kx

            jy_p = idx(ix, (iy + 1) % Ny)
            jy_m = idx(ix, (iy - 1) % Ny)
            fc[i, i, 1, 1] += 2 * ky
            fc[i, jy_p, 1, 1] -= ky
            fc[i, jy_m, 1, 1] -= ky

    # 2) if no vacancy: return pristine
    if system.vacancy is None:
        return fc

    # 3) apply vacancy by removing bonds to that site
    vx, vy, _vz = system.vacancy
    iv = idx(vx, vy)

    # four neighbors of vacancy (periodic)
    in_xp = idx((vx + 1) % Nx, vy)  # right neighbor
    in_xm = idx((vx - 1) % Nx, vy)  # left neighbor
    in_yp = idx(vx, (vy + 1) % Ny)  # up neighbor
    in_ym = idx(vx, (vy - 1) % Ny)  # down neighbor

    # Each neighbor loses ONE spring to vacancy
    fc[in_xp, in_xp, 0, 0] -= kx
    fc[in_xm, in_xm, 0, 0] -= kx
    fc[in_yp, in_yp, 1, 1] -= ky
    fc[in_ym, in_ym, 1, 1] -= ky

    # 4) delete vacancy row/col
    keep = np.ones(n, dtype=bool)
    keep[iv] = False
    return fc[keep][:, keep]


@dataclass(kw_only=True, frozen=True)
class NormalModeResult:
    """Result of a normal mode calculation for a phonon system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, Nbranch)
    modes: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, natom*3,Nbranch)
    q_vals: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, dim)

    def get_mode(self, branch: int, q: tuple[float, float, float]) -> NormalMode:
        q_target = np.asarray(list(q), float)
        # nearest q
        dq = self.q_vals - q_target[None, :]
        dq -= np.rint(dq)
        iq = int(np.argmin(np.sum(dq * dq, axis=1)))
        return DefectedNormalMode(
            _system=self.system,
            omega=self.omega[iq, branch],
            modes=self.modes[iq, :, branch],
            q_val=self.q_vals[iq, :],
        )


def calculate_normal_modes(system: System) -> NormalModeResult:
    cell = PhonopyAtoms(
        symbols=system.unit_symbols,
        cell=system.unit_cell,
        scaled_positions=system.unit_scaled_positions,
    )

    supercell_matrix = np.diag(system.n_repeats)
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    phonon.force_constants = build_force_constant_matrix(system)

    phonon.run_mesh(system.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)

    mesh_dict = phonon.get_mesh_dict()

    return NormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"] * 1e12 * 2 * np.pi,
        modes=mesh_dict["eigenvectors"],
        q_vals=mesh_dict["qpoints"],
    )


class NormalMode(ABC):
    # @abstractmethod
    def get_displacement(self, time: float = 0) -> np.ndarray[Any, Any]: ...

    @property
    @abstractmethod
    def system(self) -> System: ...


@dataclass(kw_only=True, frozen=True)
class PristineNormalMode(NormalMode):
    _system: System
    omega: float
    """Frequency of one mode (rad/s)."""
    modes: np.ndarray
    """Eigenvector of that mode."""
    q_val: np.ndarray
    """Wave vector for this mode."""

    @property
    @override
    def system(self) -> System:
        return self._system

    def get_displacement(self, time: float = 0.0) -> np.ndarray[Any, Any]:
        system = self.system
        nx, ny, nz = system.n_repeats
        qx, qy, qz = self.q_val
        print("Calling Pristine Mode")
        # phase(i,j) = exp(2πi (qx*i/Nx + qy*j/Ny) - i ω t)
        phx = np.exp(2j * np.pi * qx * (np.arange(nx) / nx))  # (Nx,)
        phy = np.exp(2j * np.pi * qy * (np.arange(ny) / ny))  # (Ny,)
        phz = np.exp(2j * np.pi * qz * (np.arange(nz) / nz))  # (Ny,)
        print(phx.shape)

        phase = (
            phx[:, None, None]
            * phy[None, :, None]
            * phz[None, None, :]
            * np.exp(-1j * self.omega * time)
        )  # (Nx,Ny)
        print(phase.shape)
        return np.real(phase[..., None] * self.modes)  # (Nx,Ny,3)


@dataclass(kw_only=True, frozen=True)
class DefectedNormalMode(NormalMode):
    _system: System
    omega: float
    """Frequency of one mode (rad/s)."""
    modes: np.ndarray
    """Eigenvector of that mode."""
    q_val: np.ndarray
    """Wave vector for this mode."""

    @property
    @override
    def system(self) -> System:
        return self._system

    def get_displacement(self, time: float = 0.0) -> np.ndarray[Any, Any]:
        system = self.system
        nx, ny, nz = system.Defected_cell_size
        qx, qy, qz = self.q_val

        # 1) build occupancy mask
        occ = np.ones((nx, ny, nz), dtype=bool)
        vx, vy, vz = system.vacancy
        occ[vx % nx, vy % ny, vz % nz] = False

        # 2) all lattice indices (vectorized)
        gx, gy, gz = np.indices((nx, ny, nz))
        gx = gx[occ]
        gy = gy[occ]
        gz = gz[occ]
        # now shape: (n_atom,)
        len(gx)
        len(gy)
        evec = self.modes
        evec = evec.reshape(8, 3)
        print(evec.shape)
        # 3) phase factor per atom
        phase = np.exp(2j * np.pi * (qx * (gx / nx) + qy * (gy / ny) + qz * (gz / nz)))
        return np.real(phase[..., None] * evec)
