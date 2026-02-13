import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.defect import VacancyDefect, VacancySystem
from phonon_lifetime.modes import (
    animate_mode_2d_xy,
    plot_mode_2d_xy,
)
from phonon_lifetime.pristine import PristineSystem

if __name__ == "__main__":
    pristine = PristineSystem(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(3, 3, 1),
        spring_constant=(1, 1, 0),
    )

    vacancy_system = VacancySystem(
        pristine=pristine,
        defect=VacancyDefect(defects=[]),
    )
    mode = vacancy_system.get_mode(branch=10, q=(0, 0, 0))

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

    vacancy_system = VacancySystem(
        pristine=pristine,
        defect=VacancyDefect(defects=[0]),
    )
    mode = vacancy_system.get_mode(branch=10, q=(0, 0, 0))

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
