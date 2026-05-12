import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.Normal_Mode_2 import (
    System,
    _build_force_constant_matrix,
    calculate_normal_modes,
)


def plot_branch_frequency_scatter(
    *,
    # element: str,
    # cell: np.ndarray,
    # n_repeats: tuple[int, int, int],
    # spring_constant: tuple[float, float, float],
    system,
    branch: int,
    save_path: str,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """
    Scatter plot of eigenfrequency omega for a given phonon branch.
    Coordinates are (q_x * N_x, q_y * N_y).
    """
    # ----------------------
    # system + normal modes
    # ----------------------
    # system = System(
    #     element=element,
    #     cell=cell,
    #     n_repeats=n_repeats,
    #     spring_constant=spring_constant,
    # )

    result = calculate_normal_modes(system)
    modes_b = result.get_modes_at_branch(branch)

    omega = modes_b.omega.reshape(15, 15)  # (Nq,)
    q_vals = modes_b.q_vals.reshape(15, 15, -1)  # (Nq, 3)
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

    # ax.set_xticks(np.arange(np.floor(x.min()), np.ceil(x.max()) + 1))
    # ax.set_yticks(np.arange(np.floor(y.min()), np.ceil(y.max()) + 1))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    _Nx, _Ny = n_repeats[0], n_repeats[1]


def plot_branch_frequency_slice_qx(
    *,
    element: str,
    cell: np.ndarray,
    n_repeats: tuple[int, int, int],
    spring_constant: tuple[float, float, float],
    branch: int,
    qx_target: float = 0.0,
    tol: float = 1e-8,
    save_path: str,
    figsize: tuple[float, float] = (6, 4),
) -> None:
    """
    Plot omega(q) slice at fixed qx = qx_target for a given branch.
    Uses mesh points satisfying |q_x - qx_target| < tol.
    """
    system = System(
        element=element,
        cell=cell,
        n_repeats=n_repeats,
        spring_constant=spring_constant,
    )

    result = calculate_normal_modes(system)
    modes_b = result.get_modes_at_branch(branch)

    omega = modes_b.omega  # (Nq,)
    q_vals = modes_b.q_vals  # (Nq, 3)

    _Nx, Ny = n_repeats[0], n_repeats[1]

    # select qx (qy) slice

    mask = np.abs(q_vals[:, 0] - qx_target) < tol

    qy = q_vals[mask, 1] * Ny
    omega_slice = omega[mask]

    # sort by qy (purely for nicer plotting)
    order = np.argsort(qy)
    qy = qy[order]
    omega_slice = omega_slice[order]

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(qy, omega_slice, "o-", ms=5)

    ax.set_xlabel(r"$q_y N_y$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(rf"Branch {branch}: frequency slice at $q_x = {qx_target}$")

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_branch_frequency_scatter(
        element="Ni",
        cell=np.diag([1.0, 2.0, 1.0]),
        n_repeats=(15, 15, 1),
        spring_constant=(1.0, 1.0, 0.0),
        branch=1,
        save_path="./examples/2D_Results/Eigenfrequencies/Scattering Plot of Frequency, y Branch.png",
    )
    plot_branch_frequency_slice_qx(
        element="Ni",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(15, 15, 1),
        spring_constant=(1.0, 1.0, 0.0),
        branch=1,
        qx_target=0.0,
        save_path="./examples/2D_Results/Eigenfrequencies/Frequency_slice_qx0.png",
    )
    FC = _build_force_constant_matrix(
        System(
            element="Ni",
            cell=np.diag([1.0, 1.0, 1.0]),
            n_repeats=(3, 3, 1),
            spring_constant=(1.0, 1.0, 0.0),
        )
    )

    print(FC[:, :, 1, 1])
