from __future__ import annotations

import numpy as np

from phonon_lifetime.Normal_Mode_2 import (
    System,
    calculate_normal_modes,
)


def test_dispersion() -> None:
    chain = System(
        element="Au",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(15, 1, 1),
        spring_constant=(1, 0.0, 0.0),
    )
    N = chain.n_repeats[0]
    m = chain.mass
    modes = calculate_normal_modes(chain)
    dispersion = modes.omega[:, 2]
    a = 1.0
    q = 2 * np.pi * (np.arange(N) - (N - 1) / 2) / N
    K = chain.spring_constant[0]
    a = chain.cell[0, 0]
    theory_dispersion = 2 * np.sqrt(K / m) * np.abs(np.sin(q * a / 2)) * 1e12 * 98.22
    np.testing.assert_allclose(dispersion, theory_dispersion, rtol=0.001)
