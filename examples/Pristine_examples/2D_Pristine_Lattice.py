from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.Mass_Defect_Phonon import System, calculate_normal_modes


def centers_to_edges(x):
    x = np.asarray(x)
    dx = np.diff(x)
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


def plot_2D_dispersion_mesh_correct() -> None:
    Nx, Ny = 11, 11
    branch = 0

    system = System(
        element="Ni",
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(Nx, Ny, 1),
        spring_constant=(1.0, 1.0, 0.0),
    )

    result = calculate_normal_modes(system)

    q_vals = result.q_vals
    omega = result.omega[:, branch]

    qx = q_vals[:, 0] * 2 * np.pi
    qy = q_vals[:, 1] * 2 * np.pi

    qx_round = np.round(qx, 12)
    qy_round = np.round(qy, 12)

    qx_unique = np.sort(np.unique(qx_round))
    qy_unique = np.sort(np.unique(qy_round))

    omega_grid = np.full((len(qy_unique), len(qx_unique)), np.nan)

    for x, y, w in zip(qx_round, qy_round, omega, strict=False):
        ix = np.where(qx_unique == x)[0][0]
        iy = np.where(qy_unique == y)[0][0]
        omega_grid[iy, ix] = w

    qx_edges = centers_to_edges(qx_unique)
    qy_edges = centers_to_edges(qy_unique)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.pcolormesh(
        qx_edges,
        qy_edges,
        omega_grid,
        shading="flat",
        cmap="viridis",
    )

    ax.set_xlabel(r"$q_x$", fontsize=12)
    ax.set_ylabel(r"$q_y$", fontsize=12)
    ax.set_title(
        r"2D phonon dispersion (Branch 0)" + "\n" + r"$\omega(q_x,q_y)$",
        fontsize=13,
    )

    ax.set_aspect("equal")
    ax.grid(False)

    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(r"$\omega$", fontsize=12)

    plt.tight_layout()

    plot_output = Path(
        "examples/Pristine_examples/2D_Pristine_Dispersion_mesh_correct.png"
    )
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output, dpi=300, bbox_inches="tight")

    print(plot_output)
    plt.show()


if __name__ == "__main__":
    plot_2D_dispersion_mesh_correct()
