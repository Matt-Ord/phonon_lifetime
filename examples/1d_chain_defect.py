import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.defect import (
    MassDefect,
    MassDefectSystem,
    VacancyDefect,
    VacancySystem,
)
from phonon_lifetime.modes import (
    animate_mode_1d_x,
    plot_mode_1d_x,
)
from phonon_lifetime.pristine import PristineSystem

if __name__ == "__main__":
    system = PristineSystem(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(101, 1, 1),
        spring_constant=(1, 0.0, 0.0),
    )

    vacancy_system = VacancySystem(
        pristine=system,
        defect=VacancyDefect(defects=[]),
    )
    mode = vacancy_system.get_mode(branch=203, q=(0, 0, 0))
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

    mass_defect_system = MassDefectSystem(
        pristine=system,
        defect=MassDefect(defects=[(1, 0)]),
    )
    # Branch 203 has the 0 atom stationary
    # Branch 204 the 0 mode moves, and this is a test of
    # us properly rescaling the mode displacements by the mass
    mode = mass_defect_system.get_mode(branch=204, q=(0, 0, 0))
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
