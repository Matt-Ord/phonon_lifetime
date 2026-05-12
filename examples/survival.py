import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.defect import (
    MassDefect,
    as_pristine_gamma_phonons,
    with_mass_defect,
)
from phonon_lifetime.lifetimes import (
    plot_first_order_scatter,
    plot_first_order_scatter_against_qx,
    plot_overlap_weights,
    plot_survival_probability,
)
from phonon_lifetime.phonon import as_gamma_phonons, get_gamma_phonons, get_mesh_phonons
from phonon_lifetime.system import build as build_system

if __name__ == "__main__":
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, False, False), cutoff=1.1
    )
    pristine_phonons = get_mesh_phonons(system, n_repeats=(111, 1, 1))
    mode_idx = pristine_phonons.get_mode_idx(branch=2, iq=(10, 0, 0))
    pristine_phonons_gamma = as_gamma_phonons(pristine_phonons)

    fig, ax = plt.subplots()
    times = np.linspace(0, 20, 500)

    for mass in [10, 10.5, 11, 11.5, 12]:
        defect = with_mass_defect(
            pristine=pristine_phonons_gamma.system,
            defects=MassDefect(defects=[(None, mass, 0)]),
        )
        defect_phonons = get_gamma_phonons(defect)
        defect_phonons = as_pristine_gamma_phonons(defect_phonons)

        _, _, line = plot_survival_probability(
            pristine_phonons_gamma[mode_idx], defect_phonons, times=times, ax=ax
        )
        line.set_label(f"Mass {mass}")

    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig("./examples/figures/survival.against_mass.png", dpi=300)

    # If we plot the rate against time, it eventually converges to a constant value.
    defect = with_mass_defect(
        pristine=pristine_phonons_gamma.system,
        defects=MassDefect(defects=[(None, 5, 0)]),
    )
    defect_phonons = get_gamma_phonons(defect)
    defect_phonons = as_pristine_gamma_phonons(defect_phonons)

    fig.savefig("./examples/figures/survival.rate_against_time.png", dpi=300)

    fig, ax, line = plot_overlap_weights(
        pristine_phonons_gamma[mode_idx],
        defect_phonons,
    )
    ax.set_title("Overlap Weights of Defect Modes")
    fig.savefig("./examples/figures/survival.overlap_weights.png", dpi=300)

    fig, ax, line = plot_first_order_scatter(
        pristine_phonons_gamma, defect_phonons, pristine_idx=mode_idx
    )
    ax.set_title("First-order Scattering of Defect Modes")
    fig.savefig("./examples/figures/survival.first_order_scatter.png", dpi=300)

    fig, ax, line = plot_first_order_scatter_against_qx(
        pristine_phonons, defect_phonons, pristine_idx=mode_idx
    )
    ax.set_title("First-order Scattering of Defect Modes against qx")
    fig.savefig("./examples/figures/survival.first_order_scatter_qx.png", dpi=300)
