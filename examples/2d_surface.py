import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime import System
from phonon_lifetime.modes import (
    animate_mode_2d_xy,
    calculate_normal_modes,
    plot_dispersion_2d_xy,
    plot_mode_2d_xy,
)

if __name__ == "__main__":
    system = System(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(11, 11, 1),
        spring_constant=(1, 1, 0),
    )
    result = calculate_normal_modes(system)

    mode = result.get_mode(branch=2, q=(1, 0, 0))
    fig, ax, _ = plot_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.mode.png", dpi=300)

    fig, ax, anim = animate_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.mode_animation.gif", dpi=300, writer="pillow"
    )

    fig, ax = plt.subplots()
    fig, ax, mesh = plot_dispersion_2d_xy(result, branch=2, ax=ax)
    fig.colorbar(mesh, label="Energy (THz)")

    ax.set_title("Phonon Dispersion Relation for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.dispersion.2.png", dpi=300)

    fig, ax = plt.subplots()
    fig, ax, mesh = plot_dispersion_2d_xy(result, branch=1, ax=ax)
    fig.colorbar(mesh, label="Energy (THz)")

    ax.set_title("Phonon Dispersion Relation for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.dispersion.1.png", dpi=300)
