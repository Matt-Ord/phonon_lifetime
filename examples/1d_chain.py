import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime import System
from phonon_lifetime.modes import (
    calculate_normal_modes,
    plot_1d_dispersion,
    plot_mode_1d_x,
)

if __name__ == "__main__":
    system = System(
        element="Ni",
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(5, 1, 1),
        spring_constant=(1, 0.0, 0.0),
    )
    result = calculate_normal_modes(system)

    mode = result.get_mode(branch=2, q=(1, 0, 0))
    fig, ax = plot_mode_1d_x(mode)
    ax.set_title("Phonon Mode for 1D Chain")
    plt.savefig("./examples/1d_chain.mode.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    fig, ax, line = plot_1d_dispersion(result, branch=0, ax=ax)
    line.set_label("Branch 0")
    fig, ax, line = plot_1d_dispersion(result, branch=1, ax=ax)
    line.set_label("Branch 1")
    fig, ax, line = plot_1d_dispersion(result, branch=2, ax=ax)
    line.set_label("Branch 2")
    ax.legend()
    ax.set_title("Phonon Dispersion Relation for 1D Chain")
    plt.savefig("./examples/1d_chain.dispersion.png", dpi=300)
    plt.close()
