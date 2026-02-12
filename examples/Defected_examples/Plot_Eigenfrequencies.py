import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.Defected_Normal_Mode import (
    System,
    calculate_normal_modes,
)


def plot_branch_frequency_scatter(
    *,
    system,
    branch: int,
    save_path: str,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    result = calculate_normal_modes(system)
    modes_b = result.get_modes_at_branch(branch)

    omega = modes_b.omega.reshape(3, 3)  # (Nq,)
    q_vals = modes_b.q_vals.reshape(3, 3, -1)  # (Nq, 3)
    q_vals = np.fft.fftshift(q_vals, axes=(0, 1))
    omega = np.fft.fftshift(omega, axes=(0, 1))

    # scatter coordinates

    x = q_vals[:, :, 0]
    y = q_vals[:, :, 1]

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.pcolormesh(x, y, omega, cmap="viridis")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\omega$")

    ax.set_xlabel(r"$q_x N_x$")
    ax.set_ylabel(r"$q_y N_y$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Branch {branch}: eigenfrequency scatter")

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


plot_branch_frequency_scatter(
    system=System(
        cell=np.diag([3.0, 3.0, 3.0]),
        spring_constant=(1.0, 1.0, 0.0),
        symbols=["Ni"] * 8,  # 8 symbols
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1 / 3, 0.0, 0.0],
            [2 / 3, 0.0, 0.0],
            [0.0, 1 / 3, 0.0],
            # Vacancy
            [2 / 3, 1 / 3, 0.0],
            [0.0, 2 / 3, 0.0],
            [1 / 3, 2 / 3, 0.0],
            [2 / 3, 2 / 3, 0.0],
        ],
    ),
    branch=1,
    save_path="./examples/Defected_examples/Eigenfrequency_of_branch_1.png",
)
