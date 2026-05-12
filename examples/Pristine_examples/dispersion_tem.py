from __future__ import annotations

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.normal_modes_lifetime import (
    System,
    calculate_normal_modes,
)

if __name__ == "__main__":
    chain = System(
        element="Ni",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(15, 1, 1),
        spring_constant=(1, 1.0, 0.0),
    )
    N = chain.n_repeats[0]
    modes = calculate_normal_modes(chain)
    print(modes.omega[:, 2])
    m = chain.mass
    print(m)
    a = 1.0
    q = 2 * np.pi * (np.arange(N) - (N - 1) / 2) / N
    K = chain.spring_constant[1]
    a = chain.cell[0, 0]
    theory_dispersion = 4 * np.sqrt(K) * np.abs(np.sin(q * a / 2)) * 1e12 * 2 * pi

    plot_output = "1d_chain_thy_dispersion.png"
    fig, ax = plt.subplots()
    ax.plot(q, theory_dispersion)
    fig.savefig(plot_output)
