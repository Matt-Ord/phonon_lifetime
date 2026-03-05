import matplotlib.pyplot as plt

from phonon_lifetime import pristine
from phonon_lifetime.modes import (
    animate_mode_1d_x,
    plot_mode_1d_x,
)
from phonon_lifetime.pristine import plot_dispersion_1d
from phonon_lifetime.system import build

if __name__ == "__main__":
    system = build.cubic(
        mass=10, distance=1.0, n_repeats=(101, 1, 1), structure="simple"
    )
    system = pristine.with_nearest_neighbor_force(
        system, spring_constant=1.0, periodic=(True, False, False), cutoff=1.1
    )

    result = system.get_modes()
    mode = result.select_mode(branch=2, q=(1, 0, 0))
    fig, ax, _ = plot_mode_1d_x(mode)
    ax.set_title("Phonon Mode for 1D Chain")
    fig.savefig("./examples/figures/1d_chain.mode.png", dpi=300)

    fig, ax, anim = animate_mode_1d_x(mode)
    ax.set_title("Phonon Mode for 1D Chain")
    anim.save(
        "./examples/figures/1d_chain.mode_animation.gif", dpi=300, writer="pillow"
    )

    fig, ax = plt.subplots()
    fig, ax, line = plot_dispersion_1d(result.at_branch(0), ax=ax)
    line.set_label("Branch 0")
    fig, ax, line = plot_dispersion_1d(result.at_branch(1), ax=ax)
    line.set_label("Branch 1")
    fig, ax, line = plot_dispersion_1d(result.at_branch(2), ax=ax)
    line.set_label("Branch 2")
    ax.legend()
    ax.set_title("Phonon Dispersion Relation for 1D Chain")
    fig.savefig("./examples/figures/1d_chain.dispersion.png", dpi=300)
