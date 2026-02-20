import numpy as np

from phonon_lifetime.defect import VacancyDefect, VacancySystem
from phonon_lifetime.modes import (
    animate_mode_2d_xy,
    animate_mode_xyz,
    plot_mode_2d_xy,
    repeat_mode,
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
    mode = vacancy_system.get_mode(idx=10)

    fig, ax, _ = plot_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.defect.mode.0.png", dpi=300)

    fig, ax, anim = animate_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.0.gif",
        dpi=300,
        writer="pillow",
    )

    vacancy_system = VacancySystem(
        pristine=pristine,
        defect=VacancyDefect(defects=[1]),
    )
    mode = vacancy_system.get_mode(idx=10)

    fig, ax, _ = plot_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.defect.mode.1.png", dpi=300)

    fig, ax, anim = animate_mode_2d_xy(mode)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.1.gif",
        dpi=300,
        writer="pillow",
    )

    fig, ax, anim = animate_mode_xyz(repeat_mode(mode, n_repeats=(3, 3, 1)))
    ax.view_init(elev=90, azim=0)  # View from above (90 degrees above the plane)
    anim.save(
        "./examples/figures/2d_surface.defect.mode_3d_animation.above.1.gif",
        dpi=300,
        writer="pillow",
    )
