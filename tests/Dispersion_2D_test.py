from __future__ import annotations

import numpy as np

from phonon_lifetime.Normal_Mode_2 import (
    System,
    calculate_normal_modes,
)


def test_dispersion_2d_square_lattice() -> None:
    # 2D square lattice toy model (monoatomic, nearest-neighbor springs)
    sheet = System(
        element="Au",
        cell=np.diag([1.0, 1.0, 10.0]),  # big c to mimic 2D
        n_repeats=(15, 13, 1),  # (Nx, Ny, 1)
        spring_constant=(1.2, 0.7, 0.0),  # (Kx, Ky, 0)
    )

    m = sheet.mass
    kx, ky = sheet.spring_constant[0], sheet.spring_constant[1]

    modes = calculate_normal_modes(sheet)  # q_vals shape: (Nq, 3), omega: (Nq, 3)

    # ---------- Γ -> X : (qx, 0, 0) ----------
    mask_gx = np.isclose(modes.q_vals[:, 1], 0.0) & np.isclose(modes.q_vals[:, 2], 0.0)
    qx_frac = modes.q_vals[mask_gx, 0]
    omega_gx = modes.omega[mask_gx]  # (Nline, 3)

    # sort along the line by qx
    sort_idx = np.argsort(qx_frac)
    qx_frac = qx_frac[sort_idx]
    omega_gx = omega_gx[sort_idx]

    # In this toy model, only x-polarized branch disperses along Γ-X; y and z are (near) zero.
    omega_gx_num = np.max(omega_gx, axis=1)

    # theory: ωx(qx,0) = 2*sqrt(kx/m)*|sin(qx*a/2)|
    # with qx_phys = (2π/a)*qx_frac  =>  qx_phys*a/2 = π*qx_frac
    omega_gx_th = 2 * np.sqrt(kx / m) * np.abs(np.sin(np.pi * qx_frac)) * 1e12 * 98.22

    np.testing.assert_allclose(omega_gx_num, omega_gx_th, rtol=1e-3, atol=0.0)

    # ---------- Γ -> Y : (0, qy, 0) ----------
    mask_gy = np.isclose(modes.q_vals[:, 0], 0.0) & np.isclose(modes.q_vals[:, 2], 0.0)
    qy_frac = modes.q_vals[mask_gy, 1]
    omega_gy = modes.omega[mask_gy]

    sort_idx = np.argsort(qy_frac)
    qy_frac = qy_frac[sort_idx]
    omega_gy = omega_gy[sort_idx]

    # only y-polarized branch disperses along Γ-Y
    omega_gy_num = np.max(omega_gy, axis=1)
    omega_gy_th = 2 * np.sqrt(ky / m) * np.abs(np.sin(np.pi * qy_frac)) * 1e12 * 98.22

    np.testing.assert_allclose(omega_gy_num, omega_gy_th, rtol=1e-3, atol=0.0)
