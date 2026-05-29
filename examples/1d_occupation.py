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
    animate_phonon_1d_x,
    get_mesh_phonons,
)
from phonon_lifetime.system import build as build_system

if __name__ == "__main__":
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        SuperCell(cell, n_repeats=(5, 1, 1)),
        spring_constant=1.0,
        periodic=(True, False, False),
        threshold=(0.9, 1.1),
    ) + build_system.with_nearest_neighbor_forces(
        SuperCell(cell, n_repeats=(5, 1, 1)),
        spring_constant=0.5,
        periodic=(True, False, False),
        threshold=(1.9, 2.1),
    )

    locations = [0]
    symmetry_broken_system = with_mass_defect(
        pristine=system,
        defects=MassDefect(
            defects=[(None, 110.01, location) for location in locations]
        ),
    )
    pristine_phonons = get_mesh_phonons(symmetry_broken_system, (3, 1, 1))
    pristine_phonons = as_pristine_gamma_phonons(pristine_phonons)
    mode_idx = np.argsort(pristine_phonons.omega)[43]

    fig, ax, anim = animate_phonon_1d_x(pristine_phonons[mode_idx])
    ax.set_title("Phonon Mode with Broken Symmetry")
    anim.save(
        "./examples/figures/1d_occupation.mode_animation.symmetry_broken.gif",
        dpi=300,
        writer="pillow",
    )

    fig, ax = plt.subplots()
    times = np.linspace(0, 10, 500)

    for mass in [10.001, 20, 30, 40, 50]:
        defect = with_mass_defect(
            pristine=system,
            defects=MassDefect(
                defects=[(None, mass, location) for location in locations]
            ),
        )
        defect_phonons = get_mesh_phonons(defect, (33, 1, 1))
        defect_phonons = as_pristine_gamma_phonons(defect_phonons)

        _, _, line = plot_survival_probability(
            pristine_phonons[mode_idx], defect_phonons, times=times, ax=ax
        )
        line.set_label(f"Mass {mass}")

    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig("./examples/figures/1d_occupation.survival.against_mass.png", dpi=300)

    fig, ax, line = plot_first_order_scatter(
        pristine_phonons, defect_phonons, pristine_idx=mode_idx
    )
    ax.set_title("First-order Scattering of Defect Modes")
    fig.savefig("./examples/figures/1d_occupation.first_order_scatter.png", dpi=300)
