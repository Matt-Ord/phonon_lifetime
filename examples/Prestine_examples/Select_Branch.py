from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.Normal_Mode_2 import System, calculate_normal_modes


def _polarization_projections(
    modes_q: np.ndarray,  # (n_dof, n_branch) where n_dof = n_atom*3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Px, Py, Pz) each of shape (n_branch,) at fixed q."""
    n_dof, n_branch = modes_q.shape
    if n_dof % 3 != 0:
        msg = f"Expected n_dof multiple of 3, got {n_dof}"
        raise ValueError(msg)
    n_atom = n_dof // 3

    e = modes_q.reshape(n_atom, 3, n_branch)  # (n_atom, 3, n_branch)
    e2 = np.abs(e) ** 2  # allow complex eigenvectors
    Px = e2[:, 0, :].sum(axis=0)
    Py = e2[:, 1, :].sum(axis=0)
    Pz = e2[:, 2, :].sum(axis=0)
    return Px, Py, Pz


def _select_branch_by_polarization(
    omega_q: np.ndarray,  # (n_branch,)
    modes_q: np.ndarray,  # (n_dof, n_branch)
    pol_axis: str = "x",  # "x"|"y"|"z"
    k_candidates: int = 6,  # only search among the lowest-k by omega
) -> int:
    """Pick branch index maximizing P_axis within the lowest-k omega branches."""
    pol_axis = pol_axis.lower()
    if pol_axis not in {"x", "y", "z"}:
        msg = "pol_axis must be one of: 'x', 'y', 'z'"
        raise ValueError(msg)

    n_branch = omega_q.shape[0]
    k = min(max(int(k_candidates), 1), n_branch)

    order = np.argsort(omega_q)  # ascending omega
    cand = order[:k]

    Px, Py, Pz = _polarization_projections(modes_q)
    P = {"x": Px, "y": Py, "z": Pz}[pol_axis]

    best = cand[np.argmax(P[cand])]
    return int(best)


def plot_scatter_and_slice_dispersion(
    *,
    Nx: int = 15,
    Ny: int = 15,
    pol_axis: str = "x",
    k_candidates: int = 6,
    cmap: str = "viridis",
    slice_qx: float = 0.0,  # slice in fractional qx
    slice_tol: float = 1e-10,  # tolerance for selecting qx ~ slice_qx
    out_scatter: str = "./examples/omega_scatter.png",
    out_slice: str = "./examples/omega_slice_qx0.png",
) -> None:
    """
    1) Scatter: (qx*Nx, qy*Ny) colored by omega on a polarization-tracked branch.
    2) Slice dispersion: fix qx ~ slice_qx and plot omega vs (qy*Ny).
    Both figures are saved to disk (VSCode-friendly).
    """
    # -------------------------------
    # Build system (same as before)
    # -------------------------------
    system = System(
        element="Ni",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(Nx, Ny, 1),
        spring_constant=(1.0, 1.0, 0.0),
    )

    result = calculate_normal_modes(system)

    q_vals = result.q_vals  # (Nq, 3)
    omega_all = result.omega  # (Nq, n_branch)
    modes_all = result.modes  # (Nq, n_dof, n_branch)

    qxN = q_vals[:, 0] * Nx
    qyN = q_vals[:, 1] * Ny

    Nq, _n_branch = omega_all.shape

    # -------------------------------
    # Select polarization-tracked branch per q
    # -------------------------------
    sel = np.empty(Nq, dtype=int)
    for iq in range(Nq):
        sel[iq] = _select_branch_by_polarization(
            omega_q=omega_all[iq],
            modes_q=modes_all[iq],
            pol_axis=pol_axis,
            k_candidates=k_candidates,
        )
    W = omega_all[np.arange(Nq), sel]  # omega on tracked branch

    # -------------------------------
    # 1) Scatter plot (save)
    # -------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sc = ax1.scatter(qxN, qyN, c=W, s=25, cmap=cmap)

    cbar = fig1.colorbar(sc, ax=ax1)
    cbar.set_label(r"$\omega$")

    ax1.set_xlabel(r"$q_x N_x$")
    ax1.set_ylabel(r"$q_y N_y$")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        rf"Polarization-tracked '{pol_axis}' branch (lowest {k_candidates} candidates)"
    )

    ax1.set_xticks(np.arange(np.floor(qxN.min()), np.ceil(qxN.max()) + 1, 1))
    ax1.set_yticks(np.arange(np.floor(qyN.min()), np.ceil(qyN.max()) + 1, 1))

    fig1.tight_layout()
    fig1.savefig(out_scatter, dpi=300)
    plt.close(fig1)

    # -------------------------------
    # 2) Slice: qx ~ slice_qx (save)
    # -------------------------------
    # Select q points with qx close to slice_qx (fractional coordinate)
    mask = np.isclose(q_vals[:, 0], slice_qx, atol=slice_tol)

    if not np.any(mask):
        msg = (
            f"No q-points found with qx ~ {slice_qx} (tol={slice_tol}). "
            f"Try increasing slice_tol (e.g. 1e-6) or choose another slice_qx."
        )
        raise RuntimeError(msg)

    qyN_slice = qyN[mask]
    omega_slice = W[mask]

    # For a clean dispersion curve, sort by qyN
    order = np.argsort(qyN_slice)
    qyN_slice = qyN_slice[order]
    omega_slice = omega_slice[order]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(qyN_slice, omega_slice, marker="o", linewidth=1.0, markersize=3)

    ax2.set_xlabel(r"$q_y N_y$  (slice at $q_x \approx$ " + f"{slice_qx}" + r")")
    ax2.set_ylabel(r"$\omega$")
    ax2.grid(True, alpha=0.3)
    ax2.set_title(rf"Dispersion slice: tracked '{pol_axis}' branch")

    fig2.tight_layout()
    fig2.savefig(out_slice, dpi=300)
    plt.close(fig2)

    print(f"Saved scatter to: {out_scatter}")
    print(f"Saved slice   to: {out_slice}")


if __name__ == "__main__":
    plot_scatter_and_slice_dispersion(
        Nx=15,
        Ny=15,
        pol_axis="x",  # 改成 "y" 就追踪 y-branch
        k_candidates=6,
        slice_qx=2.0 / 15.0,
        slice_tol=1e-10,  # 如果抓不到点，把它调大到 1e-6 之类
        out_scatter="./examples/omega_scatter_xbranch.png",
        out_slice="./examples/dispersion_slice_qx0_xbranch.png",
    )
