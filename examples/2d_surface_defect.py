from phonon_lifetime.cell import build
from phonon_lifetime.defect import VacancyDefect, with_vacancy_defect
from phonon_lifetime.phonon import (
    animate_phonon_xy,
    get_gamma_phonons,
    plot_phonon_xy,
)
from phonon_lifetime.system import as_supercell
from phonon_lifetime.system import build as build_system

if __name__ == "__main__":
    cell = build.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, True, False), cutoff=1.1
    )

    vacancy_system = with_vacancy_defect(
        pristine=as_supercell(system, n_repeats=(3, 3, 1)),
        defects=VacancyDefect(defects=[0]),
    )
    phonon = get_gamma_phonons(vacancy_system)[11]

    fig, ax, _ = plot_phonon_xy(phonon, scale_displacement=0.2)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.defect.mode.0.png", dpi=300)

    fig, ax, anim = animate_phonon_xy(phonon, scale_displacement=0.2)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.0.gif",
        dpi=300,
        writer="pillow",
    )

    vacancy_system = with_vacancy_defect(
        pristine=as_supercell(system, n_repeats=(3, 3, 1)),
        defects=VacancyDefect(defects=[1]),
    )
    phonon = get_gamma_phonons(vacancy_system)[10]

    fig, ax, _ = plot_phonon_xy(phonon, scale_displacement=0.2)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.defect.mode.1.png", dpi=300)

    fig, ax, anim = animate_phonon_xy(phonon, scale_displacement=0.2)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.1.gif",
        dpi=300,
        writer="pillow",
    )
