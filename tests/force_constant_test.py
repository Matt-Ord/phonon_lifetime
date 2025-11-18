from __future__ import annotations

import numpy as np


def test_build_force_constant_matrix() -> None:
    fc_target = np.array([[2, -2], [-2, 2]])
    A = np.zeros((2, 2, 3, 3), dtype=float)
    # Insert fc_target at the desired location
    A[0:2, 0:2, 0, 0] = fc_target
    n_repeats = np.zeros(2, dtype=int)
    spring_constant = np.zeros((2, 1))
    n_repeats[0], n_repeats[1] = 2, 1
    spring_constant[0], spring_constant[1] = 1.0, 0
    # assert system.n_repeats[1:] == (1, 1), "Only 1D chains are supported."  # noqa: ERA001
    n_x, n_y = n_repeats[0], n_repeats[1]
    kx, ky = spring_constant[0], spring_constant[1]
    n = n_x * n_y
    fc = np.zeros((n, n, 3, 3), dtype=float)

    def idx(ix: int, iy: int) -> int:
        return ix * n_y + iy

    for ix in range(n_x):
        for iy in range(n_y):
            i = idx(ix, iy)
            # X direction neighbors
            jx_p = idx((ix + 1) % n_x, iy)
            jx_m = idx((ix - 1) % n_x, iy)
            fc[i, i, 0, 0] += 2 * kx
            fc[i, jx_p, 0, 0] += -kx
            fc[i, jx_m, 0, 0] += -kx
            # Y direction neighbors
            jy_p = idx(ix, (iy + 1) % n_y)
            jy_m = idx(ix, (iy - 1) % n_y)
            fc[i, i, 1, 1] += 2 * ky
            fc[i, jy_p, 1, 1] += -ky
            fc[i, jy_m, 1, 1] += -ky
    print("A[:, :, 0, 0] =")
    print(A[:, :, 0, 0])

    print("\nfc =")
    print(fc)

    np.testing.assert_allclose(A, fc)
