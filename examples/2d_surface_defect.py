import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime import System
from phonon_lifetime.modes import (
    animate_mode_2d_xy,
    calculate_normal_modes,
    plot_mode_2d_xy,
)

if __name__ == "__main__":
    system = System(
        element="Ni",
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(11, 11, 1),
        spring_constant=(1, 1, 0),
    )
    result = calculate_normal_modes(system, vacancy=[])

    mode = result.get_mode(branch=2 * system.n_atoms + 50, q=(0, 0, 0))

    fig, ax, _ = plot_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.defect.mode.png", dpi=300)

    fig, ax, anim = animate_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.gif",
        dpi=300,
        writer="pillow",
    )
