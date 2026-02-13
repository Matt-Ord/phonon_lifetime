import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime import System
from phonon_lifetime.modes import (
    VacancyDefect,
    animate_mode_2d_xy,
    calculate_normal_modes,
    plot_mode_2d_xy,
)

if __name__ == "__main__":
    system = System(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(3, 3, 1),
        spring_constant=(1, 1, 0),
    )
    result = calculate_normal_modes(system, defect=VacancyDefect(defects=[]))

    mode = result.get_mode(branch=10, q=(0, 0, 0))

    fig, ax, _ = plot_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.defect.mode.0.png", dpi=300)

    fig, ax, anim = animate_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.0.gif",
        dpi=300,
        writer="pillow",
    )

    system = System(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(3, 3, 1),
        spring_constant=(1, 1, 0),
    )
    result = calculate_normal_modes(system, defect=VacancyDefect(defects=[0]))
    print(result.omega)

    mode = result.get_mode(branch=10, q=(0, 0, 0))

    fig, ax, _ = plot_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    plt.savefig("./examples/figures/2d_surface.defect.mode.1.png", dpi=300)

    fig, ax, anim = animate_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.1.gif",
        dpi=300,
        writer="pillow",
    )
