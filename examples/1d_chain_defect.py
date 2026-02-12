import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime import System
from phonon_lifetime.modes import (
    MassDefect,
    VacancyDefect,
    animate_mode_1d_x,
    calculate_normal_modes,
    plot_mode_1d_x,
)

if __name__ == "__main__":
    system = System(
        element="Ni",
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(101, 1, 1),
        spring_constant=(1, 0.0, 0.0),
    )
    result = calculate_normal_modes(system, defect=VacancyDefect(defects=[]))

    mode = result.get_mode(branch=203, q=(0, 0, 0))
    fig, ax, _ = plot_mode_1d_x(mode)
    ax.set_title("Phonon Mode for 1D Chain with Vacancy Defect")
    plt.savefig("./examples/figures/1d_chain.vacancy_defect.mode.png", dpi=300)

    fig, ax, anim = animate_mode_1d_x(mode)
    ax.set_title("Phonon Mode for 1D Chain with Vacancy Defect")
    anim.save(
        "./examples/figures/1d_chain.vacancy_defect.mode_animation.gif",
        dpi=300,
        writer="pillow",
    )

    result = calculate_normal_modes(system, defect=MassDefect(defects=[("H", 0)]))

    mode = result.get_mode(branch=203, q=(0, 0, 0))
    fig, ax, _ = plot_mode_1d_x(mode)
    ax.set_title("Phonon Mode for 1D Chain with Mass Defect")
    plt.savefig("./examples/figures/1d_chain.mass_defect.mode.png", dpi=300)

    fig, ax, anim = animate_mode_1d_x(mode)
    ax.set_title("Phonon Mode for 1D Chain with Mass Defect")
    anim.save(
        "./examples/figures/1d_chain.mass_defect.mode_animation.gif",
        dpi=300,
        writer="pillow",
    )
