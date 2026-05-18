import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.cell import SuperCell
from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.defect import (
    MassDefect,
    as_pristine_gamma_phonons,
    with_mass_defect,
)
from phonon_lifetime.lifetimes import (
    plot_first_order_scatter,
    plot_survival_probability,
)
from phonon_lifetime.phonon import (
    animate_phonon_xy,
    get_gamma_phonons,
)
from phonon_lifetime.system import build as build_system

if __name__ == "__main__":
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        SuperCell(cell, n_repeats=(15, 15, 1)),
        spring_constant=1.0,
        periodic=(True, False, False),
        threshold=(0.0, 1.1),
    ) + build_system.with_nearest_neighbor_forces(
        SuperCell(cell, n_repeats=(15, 15, 1)),
        spring_constant=0.5,
        periodic=(True, False, False),
        threshold=(1.1, 2.1),
    )
    # TODO: use mesh phonons here to achieve this??
    locations = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    symmetry_broken_system = with_mass_defect(
        pristine=system,
        defects=MassDefect(defects=[(None, 10.01, location) for location in locations]),
    )
    pristine_phonons = get_gamma_phonons(symmetry_broken_system)
    pristine_phonons = as_pristine_gamma_phonons(pristine_phonons)
    mode_idx = pristine_phonons.get_mode_idx(branch=280, iq=0)

    fig, ax, anim = animate_phonon_xy(pristine_phonons[mode_idx])
    ax.set_title("Phonon Mode for 1D Chain with Vacancy Defect")
    anim.save(
        "./examples/figures/1d_chain.vacancy_defect.mode_animation1.gif",
        dpi=300,
        writer="pillow",
    )

    fig, ax = plt.subplots()
    times = np.linspace(0, 10, 500)

    for mass in [10, 20, 30, 40, 50]:
        defect = with_mass_defect(
            pristine=system,
            defects=MassDefect(
                defects=[(None, mass, location) for location in locations]
            ),
        )
        defect_phonons = get_gamma_phonons(defect)
        defect_phonons = as_pristine_gamma_phonons(defect_phonons)

        _, _, line = plot_survival_probability(
            pristine_phonons[mode_idx], defect_phonons, times=times, ax=ax
        )
        line.set_label(f"Mass {mass}")

    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig("./examples/figures/survival.against_mass.png", dpi=300)

    fig, ax, line = plot_first_order_scatter(
        pristine_phonons, defect_phonons, pristine_idx=mode_idx
    )
    ax.set_title("First-order Scattering of Defect Modes")
    fig.savefig("./examples/figures/survival.first_order_scatter.png", dpi=300)
