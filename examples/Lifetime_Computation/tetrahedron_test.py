"""
Two-part fix:

1.  delta_weights_ltm_fixed: accepts a System instead of a bare phonon object,
    builds FC internally → no more "Dynamical matrix not built" error,
    and maps weights back to q-scan order (the index bug from before).

2.  fermi_golden_rule2_fixed: uses delta_weights_ltm_fixed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh

from phonon_lifetime.Mass_Defect_Phonon import (
    NormalModeResult,
    Plot_scattering_rate,
    System,
    build_force_constant_matrix,
    calculate_normal_modes,
    run_phonon,
)

prim = np.diag([1.0, 1.0, 1.0])
k = (1.0, 1.4, 1.9)
nrep = (5, 5, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed delta_weights_ltm
# Accepts System directly so it can always build its own phonon+FC from scratch.
# Also fixes the ir_grid_points → scan-order mapping.
# ─────────────────────────────────────────────────────────────────────────────


def delta_weights_ltm_fixed(system: System, omega_i: float) -> np.ndarray:
    """
    Compute LTM integration weights at frequency omega_i (plain freq, THz).

    Returns weights in q-SCAN order, shape (Nq * num_band,),
    consistent with omega_p.reshape(-1) and M_ij.

    Fixes vs original delta_weights_ltm:
      - builds phonon + FC internally  → no "Dynamical matrix not built" error
      - maps integration_weights[iq] (ir-order) back to scan order via
        ir_grid_points, so that weights[gp*B + b] matches omega_p[gp*B + b]
    """
    phonon = run_phonon(system)
    phonon.force_constants = build_force_constant_matrix(system)

    mesh_numbers = system.n_repeats
    phonon.run_mesh(mesh_numbers, is_mesh_symmetry=False)

    ir_gps = phonon.mesh.ir_grid_points  # shape (Nq,), ir-order
    Nq = np.prod(mesh_numbers)

    thm = TetrahedronMesh(
        cell=phonon.primitive,
        frequencies=phonon.mesh.frequencies,
        mesh=np.array(mesh_numbers, dtype="int64"),
        grid_address=phonon.mesh.grid_address,
        grid_mapping_table=phonon.mesh.grid_mapping_table,
        ir_grid_points=ir_gps,
    )
    thm.set(value="I", frequency_points=[omega_i], lang="C")
    for _ in thm:
        pass

    weights_iq = thm.get_integration_weights().squeeze()  # (Nq, num_band)
    # weights_iq[iq, b] corresponds to ir_gps[iq], not scan-index iq.
    # Remap: weights_scan[gp, b] = weights_iq[iq, b]  where ir_gps[iq] == gp.
    num_band = weights_iq.shape[1]
    weights_scan = np.empty((Nq, num_band), dtype=float)
    for iq, gp in enumerate(ir_gps):
        weights_scan[gp, :] = weights_iq[iq, :]

    return weights_scan.reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed fermi_golden_rule2
# ─────────────────────────────────────────────────────────────────────────────


def fermi_golden_rule2_fixed(
    results1: NormalModeResult,  # pristine
    results2: NormalModeResult,  # defected
    band: int,
    q: list,
    system: System,  # pristine system (for LTM)
) -> np.ndarray:
    psi_p = results1.vectors
    omega_p = results1.omega.reshape(-1)
    psi_def = results2.vectors
    omega_def = results2.omega.reshape(-1)

    mode_i = results1.get_mode(band, q)
    psi_i = mode_i.vector
    omega_i = mode_i.omega

    P_im = psi_i.conj().T @ psi_def  # (Ndef,)
    P_mj = psi_def.conj().T @ psi_p  # (Ndef, Npri)

    M_ij = np.einsum(
        "m,mj,mj->j",
        P_im,
        (omega_def[:, None] - omega_p[None, :]),
        P_mj,
    )

    shift = 1e-5
    weights = delta_weights_ltm_fixed(system, (omega_i - shift) / (2 * np.pi))

    return np.abs(M_ij).reshape(-1) * weights


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: check ir_grid_points ordering
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Diagnostic: ir_grid_points vs scan order")
print("=" * 60)

sys_pri = System(
    element="Ni", primitive_cell=prim, spring_constant=k, defects=None, n_repeats=nrep
)

phonon_check = run_phonon(sys_pri)
phonon_check.force_constants = build_force_constant_matrix(sys_pri)
phonon_check.run_mesh(nrep, is_mesh_symmetry=False)

ir_gps = phonon_check.mesh.ir_grid_points
expected = np.arange(np.prod(nrep), dtype=ir_gps.dtype)
print(f"  ir_grid_points[:10] = {ir_gps[:10]}")
print(f"  expected arange[:10]= {expected[:10]}")
print(f"  ir_grid_points == arange(Nq): {np.array_equal(ir_gps, expected)}")


# ─────────────────────────────────────────────────────────────────────────────
# Run and compare
# ─────────────────────────────────────────────────────────────────────────────

res_pri = calculate_normal_modes(sys_pri)

defect_spec = [((2, 2, 0), 58)]
sys_def = System(
    element="Ni",
    primitive_cell=prim,
    spring_constant=k,
    defects=defect_spec,
    n_repeats=nrep,
)
res_def = calculate_normal_modes(sys_def)

q = [2 / 5, 0 / 5, 0]
band = 2

omega_p_flat = res_pri.omega.reshape(-1)
omega_i = res_pri.get_mode(band, q).omega

print(f"\nomega_i = {omega_i:.5f} rad")

rate = fermi_golden_rule2_fixed(res_pri, res_def, band, q, sys_pri)

nz = ~np.isclose(rate, 0, atol=np.max(rate) * 1e-4)
omega_sc = omega_p_flat[nz]
print(f"non-zero entries          : {nz.sum()}")
print(f"omega_scattered (unique)  : {np.unique(np.round(omega_sc, 4))}")
print(f"max|omega_sc - omega_i|   : {np.max(np.abs(omega_sc - omega_i)):.6f}")

fig, ax = plt.subplots(figsize=(7, 4))
Plot_scattering_rate(fig, ax, rate, res_pri, q)
ax.set_title(f"Fixed scattering rate  q={q}  band={band}")
fig.tight_layout()
outname = (
    "./examples/Lifetime_Computation/Lifetime_results/scattering_rate_ltm_fixed.png"
)
fig.savefig(outname, dpi=250, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {outname}")
