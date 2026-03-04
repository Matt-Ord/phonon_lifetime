from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


@dataclass(kw_only=True, frozen=True)
class System:
    element: str
    primitive_cell: np.ndarray
    spring_constant: tuple[float, float, float]
    # Each defect = ((ix, iy, iz), mass)
    defects: list[tuple[tuple[int, int, int], float]] | None = None

    n_repeats: tuple[int, int, int] = (1, 1, 1)

    @property
    def unit_cell(self) -> np.ndarray:
        p = np.asarray(self.primitive_cell, float)
        if self.defects is None:
            return p
        Nx, Ny, Nz = self.n_repeats

        return np.diag([Nx, Ny, Nz]) @ p

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
    def unit_scaled_positions(self) -> np.ndarray:
        if self.defects is None:
            return np.array([[0.0, 0.0, 0.0]])
        Nx, Ny, Nz = self.n_repeats
        gx, gy, gz = np.indices((Nx, Ny, Nz))
        gx = np.swapaxes(gx, 0, 1)
        gy = np.swapaxes(gy, 0, 1)
        gz = np.swapaxes(gz, 0, 1)

        g = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(float)

        g[:, 0] /= Nx
        g[:, 1] /= Ny
        g[:, 2] /= Nz
        return g

    def get_atom_centres(self) -> np.ndarray[Any, Any]:
        # build occupancy mask

        nx, ny, nz = self.n_repeats[0], self.n_repeats[1], self.n_repeats[2]
        occ = np.ones((nx, ny, nz), dtype=bool)
        gx, gy, gz = np.indices((nx, ny, nz))
        gx = gx[occ]
        gy = gy[occ]
        gz = gz[occ]

        # grid indices, vacancy removed
        a1 = self.primitive_cell[0]
        a2 = self.primitive_cell[1]
        R = gx[:, None] * a1[None, :] + gy[:, None] * a2[None, :]

        X = R[:, 0]
        Y = R[:, 1]

        return X, Y

    @property
    def unit_symbols(self) -> list[str]:
        if self.defects is None:
            return [self.element]
        return [self.element] * (
            self.n_repeats[0] * self.n_repeats[1] * self.n_repeats[2]
        )


def _build_masses_with_defects(system: System) -> np.ndarray:
    # 1) get default masses from symbols (phonopy lookup)
    cell0 = PhonopyAtoms(
        symbols=system.unit_symbols,
        cell=system.unit_cell,
        scaled_positions=system.unit_scaled_positions,
    )
    masses = np.asarray(cell0.masses, float).copy()

    # 2) apply mass defects (grid index -> atom index)
    if system.defects is None:
        return masses

    Nx, Ny, Nz = system.n_repeats

    def to_flat_index(ix: int, iy: int, iz: int) -> int:
        ix %= Nx
        iy %= Ny
        iz %= Nz
        return (iy * Nx + ix) * Nz + iz

    for (ix, iy, iz), m in system.defects:
        masses[to_flat_index(ix, iy, iz)] = float(m)

    return masses


def build_force_constant_matrix(system):
    Nx, Ny = system.n_repeats[0], system.n_repeats[1]
    kx, ky = system.spring_constant[0], system.spring_constant[1]

    def idx(ix: int, iy: int) -> int:
        return iy * Nx + ix

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

    # defect is mass-only: FC does NOT change
    return fc


def calculate_normal_modes(system: System) -> NormalModeResult:
    masses = _build_masses_with_defects(system)
    cell = PhonopyAtoms(
        symbols=system.unit_symbols,
        cell=system.unit_cell,
        scaled_positions=system.unit_scaled_positions,
        masses=masses,
    )
    if system.defects is not None:
        supercell_matrix = np.diag([1.0, 1.0, 1.0])
        mesh = [1.0, 1.0, 1.0]
    else:
        supercell_matrix = np.diag(system.n_repeats)
        mesh = system.n_repeats

    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    phonon.force_constants = build_force_constant_matrix(system)

    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    mesh_dict = phonon.get_mesh_dict()
    return NormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"] * 2 * np.pi,
        modes=mesh_dict["eigenvectors"],
        q_vals=mesh_dict["qpoints"],
        masses=masses,
    )


@dataclass(kw_only=True, frozen=True)
class NormalModeResult:
    """Result of a normal mode calculation for a phonon system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, Nbranch)
    modes: np.ndarray[Any, np.dtype[np.complexfloating]]
    q_vals: np.ndarray[Any, np.dtype[np.floating]]  # shape = (Nq, dim)
    masses: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def vectors(self) -> np.ndarray:
        # pristine case: build (Natom*3, Nband*Nq) from displacements
        if self.system.defects is None:
            q_vals = self.q_vals
            Nq = len(q_vals)
            Nband = 3

            mode0 = self.get_mode(0, q_vals[0])
            disp0 = mode0.get_displacement(time=0)
            v0 = disp0.reshape(-1)  # length = Natom*3
            Natom = v0.size

            out = np.empty((Natom, Nband * Nq), dtype=complex)

            col = 0
            for q in q_vals:
                for b in range(Nband):
                    mode = self.get_mode(b, q)
                    disp = mode.get_displacement(time=0)
                    out[:, col] = disp.reshape(-1)  # flatten to (Natom*3,)
                    col += 1

            return out

        return self.modes[0]

    def get_mode(self, branch: int, q: tuple[float, float, float]) -> NormalMode:
        q_target = np.asarray(list(q), float)
        # nearest q
        dq = self.q_vals - q_target[None, :]
        dq -= np.rint(dq)
        iq = int(np.argmin(np.sum(dq * dq, axis=1)))
        return NormalMode(
            system=self.system,
            omega=self.omega[iq, branch],
            modes=self.modes[iq, :, branch],
            q_val=self.q_vals[iq, :],
            masses=self.masses,
        )


@dataclass(kw_only=True, frozen=True)
class NormalMode:
    system: System
    omega: float
    """Frequency of one mode (rad/s)."""
    modes: np.ndarray
    """Eigenvector of that mode."""
    q_val: np.ndarray
    """Wave vector for this mode."""
    masses: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def vector(self) -> np.ndarray:
        if self.system.defects is not None:
            return self.modes
        system = self.system
        nx, ny, nz = system.n_repeats
        qx, qy, qz = self.q_val
        # phase(i,j) = exp(2πi (qx*i/Nx + qy*j/Ny) - i ω t)
        phx = np.exp(2j * np.pi * qx * (np.arange(nx)))  # (Nx,)
        phy = np.exp(2j * np.pi * qy * (np.arange(ny)))  # (Ny,)
        phz = np.exp(2j * np.pi * qz * (np.arange(nz)))  # (Ny,)
        phase = (
            phy[:, None, None] * phx[None, :, None] * phz[None, None, :]
        )  # column y first and then row x
        return (phase[..., None] * self.modes).ravel()  # (Nx,Ny,3)

    def get_displacement(self, time: float = 0.0) -> np.ndarray[Any, Any]:
        system = self.system
        nx, ny, _nz = system.n_repeats
        _qx, _qy, _qz = self.q_val
        vector = self.vector
        return vector.reshape((nx, ny, 3))
        # * masses_2d ** (-1 / 2)


def Plot_displacement(
    mode: NormalMode,
    time: float = 0,
) -> None:
    """Quiver plot of displacement field u(x,y) for one NormalMode."""
    # call displacement
    displacement = mode.get_displacement(time=time)  # (nx, ny, 3)
    X, Y = mode.system.get_atom_centres()  # (nx, ny)
    Ux = np.real(displacement[:, :, 0])  # (nx, ny)
    Uy = np.real(displacement[:, :, 1])  # (nx, ny)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, Ux, Uy, angles="xy", scale_units="xy")
    ax.set_aspect("equal")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Displacement field (quiver)")
    Ux = np.real(displacement[:, :, 0]).ravel()
    Uy = np.real(displacement[:, :, 1]).ravel()
    for x, y, ux, uy in zip(X, Y, Ux, Uy, strict=False):
        label = f"({ux:.3g}, {uy:.3g})"
        ax.text(x, y, label, fontsize=7, ha="center", va="bottom")

    ax.margins(0.2)
    # ax.set_xlim(-0.5 * a, 0.5 * a)  # padding in y
    fig.tight_layout()
    savepath = "./examples/Lifetime_Computation/Lifetime_results/quiver_vector.png"
    print(savepath)
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig, ax


def build_H_def(results1: NormalModeResult, results2: NormalModeResult) -> np.ndarray:
    pristine_states = results1.vectors
    defected_states = results2.vectors
    omega_1 = results1.omega.reshape(-1)

    omega_1 = omega_1.reshape(-1)
    omega_2 = results2.omega[0, :]
    # reshape pristine states to keep its dimension consistent with defected: (Nb=3Nq,3*Natom)
    psi = pristine_states.reshape(-1, pristine_states.shape[-1])  # (3Nq, 3)
    # H_def = sum_n omega_n |psi_n><psi_n|
    H1 = (psi * omega_1[None, :]) @ psi.conj().T  # (3Nq, 3Nq)

    omega_2 = results2.omega[0, :]
    psi = defected_states.reshape(-1, defected_states.shape[-1])  # (3Nq, Nb)
    # H_def = sum_n omega_n |psi_n><psi_n|
    H2 = (psi * omega_2[None, :]) @ psi.conj().T  # (3N, 3N)
    return H1 - H2


def get_overlap_with_mode(
    pristine_mode: NormalMode, defected_mode: NormalModeResult
) -> np.ndarray:
    """Project a selected pristine state to all defected states."""
    pristine_vector = pristine_mode.vector
    defected_vector = defected_mode.vectors
    return np.einsum("i,ij->j", np.conj(pristine_vector), defected_vector)


def fermi_golden_rule2(
    results1: NormalModeResult,  # pristine
    results2: NormalModeResult,  # defected
    band: int,
    q: tuple[float, float, float],
) -> np.ndarray:
    # reshape the pristines.modes form (Nq,3,Nb=3) to (3*Natom,3*Nq)
    psi_p = results1.vectors
    omega_p = results1.omega.reshape(-1)

    psi_def = results2.vectors  # (3*Natom,Nb=3*Nq)
    omega_def = results2.omega.reshape(-1)

    # choose initial state |psi_i^p>
    mode_i = results1.get_mode(band, q)
    psi_i = mode_i.vector  # (Natom*3,)

    # P_im = <psi_i^p | psi_m^d>  shape (3*N_atoms,)
    P_im = psi_i.conj() @ psi_def
    # P_mj = <psi_m^d | psi_j^p>  shape (3*N_atoms,3*N_atoms )
    P_mj = psi_def.conj().T @ psi_p

    # matrix elements M_ij = Σ_m P_im (ω_m^d - ω_j^p) P_mj
    # shape: (3*N_atoms,)
    M_ij = np.einsum("m,mj,mj->j", P_im, (omega_def[:, None] - omega_p[None, :]), P_mj)
    sigma = 0.1
    omega_i = mode_i.omega
    delta_E = omega_p - omega_i
    gaussian_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(delta_E**2) / (2 * sigma**2)
    )
    # scattering strength ~ |M_ij|^2
    return np.abs(M_ij) ** 2 * gaussian_delta


def fermi_golden_rule(
    H_def, results1: NormalModeResult, results2: NormalModeResult, band
) -> np.ndarray:
    """
    Use H_def to compute Fermi Golden Rule.
    Here result1 and result2 should both be pristine states.
    """
    psi_p = results1.vectors
    psi_d = results2.get_mode(band, (0, 1 / 3, 0))  # psi_d is also pristine state here
    omega_p = results1.omega.reshape(-1)
    omega_d = psi_d.omega
    psi_d = psi_d.vector
    overlaps = np.abs(psi_p.conj().T @ psi_d)  # (Nmode,)
    # need to remove the overlap with itself, otherwise it will suppress other overlaps when plotted
    remove_index = np.argmax(overlaps)
    print(remove_index)
    # implement the <psi_m^p|H_def|psi_n^p>
    psi_p = np.delete(psi_p, remove_index, axis=1)
    omega_p = np.delete(omega_p, remove_index, axis=0)
    matrix_product = np.abs(np.conj(psi_d) @ H_def @ psi_p) ** 2

    hbar = 1.0
    E_p = hbar * omega_p
    E_d = hbar * omega_d
    sigma = 0.1
    # Use Gaussian distribution to approximate delta function
    delta_E = E_p - E_d
    gaussian_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(delta_E**2) / (2 * sigma**2)
    )
    return matrix_product * gaussian_delta


def Plot_scattering_rate(fig, ax, rate, res_pri, defect) -> None:
    """Sort Omegas and scattering rates, then plot the rate-band_index curve."""
    # flatten the pristine frequencies
    Omegas = res_pri.omega.reshape(-1)
    # sort the frequencies, then sort the scattering rate with the same indices
    indices = np.argsort(Omegas)
    Omegas = Omegas[indices]
    rate = rate[indices]
    # the x axis of plotting is band index, integers from 0 to N_states
    x = np.arange(rate.shape[0])
    # print the final states omegas which are expected to be equal to the initial
    scattering_index = np.where(~np.isclose(rate, 0, atol=0.1))
    Omegas = Omegas[scattering_index]
    print(scattering_index)
    print(Omegas)
    # label: show defect position + mass
    (ix, iy, iz), m = defect[0]

    ax.plot(
        x,
        rate,
        marker="o",
        linewidth=1.2,
        markersize=3.5,
        label=f"defect@({ix},{iy},{iz}), m={m:g}",
    )

    ax.set_xlabel("Final (defected) branch index")
    ax.set_ylabel(r"Scattering rate (arb. units)")
    ax.set_title("Scattering rate from defected branch into pristine branches")
    ax.legend(frameon=False)

    return fig


def Plot_overlap(res_pri, res_def, band_q, q_pri) -> None:
    omega2 = res_def.omega.reshape(-1)
    pristine_mode = res_pri.get_mode(band_q, q_pri)
    print(pristine_mode.omega, "Selected Omega")
    Normal = omega2.shape[0]
    overlap = get_overlap_with_mode(pristine_mode, res_def)
    overlap_abs = np.abs(overlap) / Normal * 3
    indices = np.count_nonzero(np.isclose(overlap_abs, 0, atol=0.001))
    print(overlap_abs.size - indices, "non_zero overlaps")
    x = np.asarray(omega2).reshape(-1)
    y = np.asarray(overlap_abs).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y)

    ax.set_xlabel(r"$\omega2$")
    ax.set_ylabel("overlap")
    ax.set_title("Overlap vs Omega2")

    fig.tight_layout()
    fig.savefig(
        "./examples/Lifetime_Computation/Lifetime_results/overlap_vs_omega2.png",
        dpi=300,
    )
    plt.close(fig)
    print("./examples/Lifetime_Computation/Lifetime_results/overlap_vs_omega2.png")
