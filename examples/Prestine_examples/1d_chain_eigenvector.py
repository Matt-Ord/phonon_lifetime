from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.Normal_Mode_2 import (
    System,
    calculate_normal_modes,
)

if __name__ == "__main__":
    chain = System(
        element="Ni",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(38, 1, 1),
        spring_constant=(1, 0.0, 0.0),
    )
    result = calculate_normal_modes(chain)
    b2 = result.get_ModesAtBranch(branch=2)

    Omega = b2.omega
    eigen_vectors = b2.modes
    q = b2.q_vals
    q = q[5, :]
    print(q)
    t = 0.0

    Rl = np.zeros((3, 38))
    Rl = np.zeros((3, 38))
    Rl[0, :] = np.arange(38)

    q_dot_Rl = np.dot(q, Rl)

    propagation = np.exp(1j * (q_dot_Rl - Omega * t / 9.82 / 1e13))
    print(propagation.shape)
    eigen_vector = eigen_vectors[5, :]
    displacement = eigen_vector.reshape(3, 1) @ propagation.reshape(1, 38)
    print("displacement.shape:", displacement.shape)
    x = np.arange(displacement.shape[1])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        x,
        displacement[0].real,
        marker="o",
        markersize=5,
        linewidth=2,
        label="x displacement",
    )
    ax.set_title("Displacement of Normal Mode q = -0.355", fontsize=13)

    ax.set_xlabel("l (cell index)", fontsize=12)
    ax.set_ylabel("Displacement (Å)", fontsize=12)

    ax.legend(fontsize=11, loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()

    save_folder = Path("./examples")
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / "single_mode_displacement.png"

    plt.savefig(save_path, dpi=300)
    plt.close()
