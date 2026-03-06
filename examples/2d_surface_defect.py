from phonon_lifetime import pristine
from phonon_lifetime.defect import VacancyDefect, VacancySystem
from phonon_lifetime.modes import animate_mode_xy, plot_mode_xy
from phonon_lifetime.system import build

if __name__ == "__main__":
    system = build.cubic(mass=10, distance=1.0, n_repeats=(3, 3, 1), structure="simple")
    system = pristine.with_nearest_neighbor_forces(
        system, spring_constant=1.0, periodic=(True, True, False), cutoff=1.1
    )

    vacancy_system = VacancySystem(
        pristine=system,
        defect=VacancyDefect(defects=[]),
    )
    mode = vacancy_system.get_mode(idx=11)

    fig, ax, _ = plot_mode_xy(mode, bond_cutoff=1.1)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.defect.mode.0.png", dpi=300)

    fig, ax, anim = animate_mode_xy(mode, bond_cutoff=1.1)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.0.gif",
        dpi=300,
        writer="pillow",
    )

    vacancy_system = VacancySystem(
        pristine=system,
        defect=VacancyDefect(defects=[1]),
    )
    mode = vacancy_system.get_mode(idx=10)

    fig, ax, _ = plot_mode_xy(mode, bond_cutoff=1.1)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.defect.mode.1.png", dpi=300)

    fig, ax, anim = animate_mode_xy(mode, bond_cutoff=1.1)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.defect.mode_animation.1.gif",
        dpi=300,
        writer="pillow",
    )
