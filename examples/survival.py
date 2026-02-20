import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.defect import MassDefect, MassDefectSystem
from phonon_lifetime.lifetimes import (
    calculate_decay_rates,
    plot_survival_probability,
)
from phonon_lifetime.pristine import PristineSystem

if __name__ == "__main__":
    system = PristineSystem(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(101, 1, 1),
        spring_constant=(1, 0.0, 0.0),
    )

    pristine_modes = system.get_modes()
    mode = pristine_modes.select_mode(branch=2, q=(1, 0, 0))

    fig, ax = plt.subplots()
    times = np.linspace(0, 1000, 500)
    for mass in [10, 10.5, 11, 11.5, 12]:
        defect = MassDefectSystem(
            pristine=system, defect=MassDefect(defects=[(mass, 0)])
        )

        _, _, line = plot_survival_probability(
            mode, defect.get_modes(), times=times, ax=ax
        )
        line.set_label(f"Mass {mass}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig("./examples/figures/survival.against_mass.png", dpi=300)

    # If we plot the rate against time, it eventually converges to a constant value.
    defect = MassDefectSystem(pristine=system, defect=MassDefect(defects=[(mass, 0)]))
    defect_modes = defect.get_modes()
    times = np.linspace(0, 40000, 50)[1:]
    rates = [
        calculate_decay_rates(pristine_modes.at_branch(2), defect_modes, time=t)[1]
        for t in times
    ]
    fig, ax = plt.subplots()
    ax.plot(times, rates)
    ax.legend()

    fig.savefig("./examples/figures/survival.rate_against_time.png", dpi=300)
