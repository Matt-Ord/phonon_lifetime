from __future__ import annotations

import numpy as np

from phonon_lifetime.normal_modes_lifetime import (
    System,
    _build_force_constant_matrix,  # noqa: PLC2701 # type: ignore in test folder
)


def test_build_force_constant_matrix() -> None:
    fc_target = np.array([[2, -2], [-2, 2]])
    a = np.zeros((2, 2, 3, 3), dtype=float)
    a[0:2, 0:2, 0, 0] = fc_target
    # Insert fc_target at the desired location
    chain = System(
        element="Au",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(2, 1, 1),
        spring_constant=(1, 1.0, 0.0),
    )
    fc = _build_force_constant_matrix(chain)
    np.testing.assert_allclose(a, fc)
