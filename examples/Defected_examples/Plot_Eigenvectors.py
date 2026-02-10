from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.Defected_Normal_Mode import (
    System,
    calculate_normal_modes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

pi = math.pi


def _supercell_cart_positions_from_system(system) -> np.ndarray:
    cell = np.asarray(system.cell, float)  # (3,3), row vectors
    basis_spos = np.asarray(system.scaled_positions, float)
    # Cartesian positions: (n_atom, 3)
    return basis_spos @ cell


pi = np.pi


def plot_eigenvectors(
    system,
    *,
    branch: int | None = 0,  # None -> sum over all branches
    q: Iterable[float] = (0.0, 0.0, 0.0),
    save_path: str | Path | None = None,
    dpi: int = 300,
    normalize: bool = True,  # normalize summed displacement by n_branch
) -> dict[str, Any]:
    result = calculate_normal_modes(system)

    q_target = np.asarray(list(q), float).reshape(-1)

    # nearest q
    dq = result.q_vals - q_target[None, :]
    dq -= np.rint(dq)
    iq = int(np.argmin(np.sum(dq * dq, axis=1)))
    q_mesh = result.q_vals[iq]

    # positions
    R = _supercell_cart_positions_from_system(system)

    # common Bloch phase at t=0
    phase = np.exp(1j * 2 * pi * (R @ q_mesh))

    # If branch=None, it will give sum over all branches
    if branch is None:
        # modes: (Nq, Ndof, Nbranch)
        Ndof = result.modes.shape[1]
        n_atom = Ndof // 3
        n_branch = result.modes.shape[2]

        u_sum = np.zeros((n_atom, 3), dtype=float)
        result.omega[iq].astype(float)  # (Nbranch,)
        omega_all = result.omega[iq].astype(float)
        for s in range(n_branch):
            evec = np.asarray(result.modes[iq, :, s])  # (Ndof,)
            evec_atoms = evec.reshape(n_atom, 3)
            u_real_s = np.real(evec_atoms * phase[:, None])
            u_sum += u_real_s
        u_plot = u_sum
        omega_info = float(np.mean(omega_all))  # just a summary number for title
        title = f"u_sum(t=0), all branches, iq={iq}, q={q_mesh}"
        out_name = f"evec_ALLbranches_iq{iq}.png"

        return_omega = omega_info
        return_evec = None

    else:
        # pick branch (your original behavior)
        b = result.get_modes_at_branch(branch=branch)

        omega = float(b.omega[iq])
        evec = np.asarray(b.modes[iq])  # (Ndof,)

        n_atom = evec.size // 3
        evec_atoms = evec.reshape(n_atom, 3)

        u_plot = np.real(evec_atoms * phase[:, None])
        title = f"u(t=0), branch={branch}, iq={iq}, omega={omega:.6g}"
        out_name = f"evec_branch{branch}_iq{iq}.png"

        return_omega = omega
        return_evec = evec

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(
        R[:, 0],
        R[:, 1],
        u_plot[:, 0],
        u_plot[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
    )
    ax.margins(0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()

    out_path = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        out_path = save_path / out_name
        fig.savefig(out_path, dpi=dpi)

    return {
        "iq": iq,
        "q_mesh": q_mesh,
        "omega": return_omega,
        "eigen_vector": return_evec,  # None if branch=None
        "u_real": u_plot,
        "save_path": out_path,
    }


out = plot_eigenvectors(
    system=System(
        cell=np.diag([3.0, 3.0, 3.0]),
        spring_constant=(1.0, 1.0, 0.0),
        symbols=["Ni"] * 8,  # 8 symbols
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1 / 3, 0.0, 0.0],
            [2 / 3, 0.0, 0.0],
            [0.0, 1 / 3, 0.0],
            [1 / 3, 1 / 3, 0.0],
            [2 / 3, 1 / 3, 0.0],
            [0.0, 2 / 3, 0.0],
            [1 / 3, 2 / 3, 0.0],
            [2 / 3, 2 / 3, 0.0],
        ],
    ),
    branch=5,
    q=(1 / 3, 1 / 3, 0),
    save_path="./examples/Defected_examples/Eigenvector",
)
print(out["save_path"], out["omega"])
