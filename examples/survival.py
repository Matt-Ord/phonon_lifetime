from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.defect import (
    MassDefect,
    as_pristine_gamma_phonons,
    with_mass_defect,
)
from phonon_lifetime.lifetimes import (
    calculate_decay_rates,
    plot_first_order_scatter,
    plot_first_order_scatter_against_qx,
    plot_overlap_weights,
    plot_survival_probability,
)
from phonon_lifetime.phonon import as_gamma_phonons, get_gamma_phonons, get_mesh_phonons
from phonon_lifetime.system import build as build_system

if TYPE_CHECKING:
    from ty_extensions import Unknown

if __name__ == "__main__":
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, False, False), cutoff=1.1
    )
    pristine_phonons = get_mesh_phonons(system, n_repeats=(25, 1, 1))
    pristine_phonons = pristine_phonons.at_branch(2)
    mode_idx = pristine_phonons.get_mode_idx(branch=0, iq=(5, 0, 0))
    pristine_phonons = as_gamma_phonons(pristine_phonons)

    fig, ax = plt.subplots()
    times = np.linspace(0, 20, 500)

    def _decay_fn(
        t: np.ndarray[tuple[int], np.dtype[np.float64]], *, rate: float
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.exp(-(rate * t))

    for mass in [10, 10.5, 11, 11.5, 12]:
        defect = with_mass_defect(
            pristine=pristine_phonons.system,
            defects=MassDefect(defects=[(None, mass, 0)]),
        )
        defect_phonons = get_gamma_phonons(defect)
        defect_phonons = as_pristine_gamma_phonons(defect_phonons)

        _, _, line = plot_survival_probability(
            pristine_phonons[mode_idx], defect_phonons, times=times, ax=ax
        )
        line.set_label(f"Mass {mass}")
        rate = calculate_decay_rates(
            pristine_phonons, defect_phonons, time=times[-1] * 10
        )[mode_idx]

        (ideal_line,) = ax.plot(times, _decay_fn(times, rate=rate), linestyle="--")
        ideal_line.set_color(line.get_color())
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig("./examples/figures/survival.against_mass.png", dpi=300)

    # If we plot the rate against time, it eventually converges to a constant value.
    defect = with_mass_defect(
        pristine=pristine_phonons.system, defects=MassDefect(defects=[(None, 50, 0)])
    )
    defect_phonons = get_gamma_phonons(defect)
    defect_phonons = as_pristine_gamma_phonons(defect_phonons)
    times: Unknown = np.linspace(0, times[-1] * 10, 50)[1:]
    rates = [
        calculate_decay_rates(pristine_phonons, defect_phonons, time=t)[mode_idx]
        for t in times
    ]
    fig, ax = plt.subplots()
    ax.plot(times, rates)
    ax.legend()

    fig.savefig("./examples/figures/survival.rate_against_time.png", dpi=300)

    fig, ax, line = plot_overlap_weights(
        pristine_phonons, defect_phonons, pristine_idx=mode_idx
    )
    ax.set_title("Overlap Weights of Defect Modes")
    fig.savefig("./examples/figures/survival.overlap_weights.png", dpi=300)

    fig, ax, line = plot_first_order_scatter(
        pristine_phonons, defect_phonons, pristine_idx=mode_idx
    )
    ax.set_title("First-order Scattering of Defect Modes")
    fig.savefig("./examples/figures/survival.first_order_scatter.png", dpi=300)

    fig, ax, line = plot_first_order_scatter_against_qx(
        pristine_phonons, defect_phonons, pristine_idx=mode_idx
    )
    ax.set_title("First-order Scattering of Defect Modes against qx")
    fig.savefig("./examples/figures/survival.first_order_scatter_qx.png", dpi=300)
