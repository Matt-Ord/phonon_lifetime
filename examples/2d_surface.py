import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime import System
from phonon_lifetime.modes import (
    calculate_normal_modes,
    plot_2d_dispersion,
    plot_mode_2d_xy,
)

if __name__ == "__main__":
    system = System(
        element="Ni",
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(10, 10, 1),
        spring_constant=(1, 1, 0),
    )
    result = calculate_normal_modes(system)

    mode = result.get_mode(branch=2, q=(1, 0, 0))
    fig, ax = plot_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.mode.png", dpi=300)

    fig, ax = plt.subplots()
    fig, ax, mesh = plot_2d_dispersion(result, branch=2, ax=ax)
    fig.colorbar(mesh, label="Energy (THz)")

    ax.set_title("Phonon Dispersion Relation for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.dispersion.2.png", dpi=300)

    fig, ax = plt.subplots()
    fig, ax, mesh = plot_2d_dispersion(result, branch=1, ax=ax)
    fig.colorbar(mesh, label="Energy (THz)")

    ax.set_title("Phonon Dispersion Relation for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.dispersion.1.png", dpi=300)
