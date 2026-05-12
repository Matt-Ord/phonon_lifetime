from phonon_lifetime.cell import SuperCell, build
from phonon_lifetime.defect import (
    MassDefect,
    VacancyDefect,
    with_mass_defect,
    with_vacancy_defect,
)
from phonon_lifetime.phonon import (
    animate_phonon_1d_x,
    as_supercell_phonon,
    get_gamma_phonon,
    get_gamma_phonons,
    plot_phonon_1d_x,
)
from phonon_lifetime.system import build as build_system

if __name__ == "__main__":
    cell = build.cubic(mass=10, distance=1.0, structure="simple")
    strain_system = build_system.with_nearest_neighbor_forces(
        SuperCell(cell, (101, 1, 1)),
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    vacancy_system = with_vacancy_defect(
        pristine=strain_system,
        defects=VacancyDefect(defects=[0, 40]),
    )
    phonons = get_gamma_phonons(vacancy_system)
    phonon = get_gamma_phonon(vacancy_system, branch=200)
    fig, ax, _ = plot_phonon_1d_x(phonon)
    ax.set_title("Phonon Mode for 1D Chain with Vacancy Defect")
    fig.savefig("./examples/figures/1d_chain.vacancy_defect.mode.png", dpi=300)

    phonon = as_supercell_phonon(phonon, n_repeats=(3, 1, 1))
    fig, ax, anim = animate_phonon_1d_x(phonon)
    ax.set_title("Phonon Mode for 1D Chain with Vacancy Defect")
    anim.save(
        "./examples/figures/1d_chain.vacancy_defect.mode_animation.gif",
        dpi=300,
        writer="pillow",
    )

    mass_defect_system = with_mass_defect(
        pristine=strain_system,
        defects=MassDefect(defects=[(None, 1, 0)]),
    )
    # Branch 203 has the 0 atom stationary
    # Branch 204 the 0 mode moves, and this is a test of
    # us properly rescaling the mode displacements by the mass
    phonon = get_gamma_phonon(mass_defect_system, branch=204)
    fig, ax, _ = plot_phonon_1d_x(phonon)
    ax.set_title("Phonon Mode for 1D Chain with Mass Defect")
    fig.savefig("./examples/figures/1d_chain.mass_defect.mode.png", dpi=300)

    fig, ax, anim = animate_phonon_1d_x(phonon)
    ax.set_title("Phonon Mode for 1D Chain with Mass Defect")
    anim.save(
        "./examples/figures/1d_chain.mass_defect.mode_animation.gif",
        dpi=300,
        writer="pillow",
    )
