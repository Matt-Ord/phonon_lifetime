from __future__ import annotations

import numpy as np

from phonon_lifetime.Normal_Mode_2 import (
    SquareLattice2DSystem,
    build_force_constants_2d,  # type: ignore in test folder
)


def test_build_force_constants_2d() -> None:
    nx = 5
    ny = 5
    N = nx * ny
    k_nn = 12.5

    def idx(ix, iy):
        return iy * nx + ix

    Kx = np.zeros((N, N), float)
    Ky = np.zeros((N, N), float)

    for iy in range(ny):
        for ix in range(nx):
            i = idx(ix, iy)
            # neighbours in x direction
            for jx, jy in [((ix + 1) % nx, iy), ((ix - 1) % nx, iy)]:
                j = idx(jx, jy)
                Kx[i, j] += -k_nn
            Kx[i, i] += 2 * k_nn
            # neighbours in y direction
            for jx, jy in [(ix, (iy + 1) % ny), (ix, (iy - 1) % ny)]:
                j = idx(jx, jy)
                Ky[i, j] += -k_nn
            Ky[i, i] += 2 * k_nn

    force_constant_target = np.zeros((N, N, 3, 3), float)
    force_constant_target[:, :, 0, 0] = Kx
    force_constant_target[:, :, 1, 1] = Ky
    # Insert fc_target at the desired location
    system = SquareLattice2DSystem(
        element="Si",  # Change to any symbol as required
        lattice_constantx=1.0,
        lattice_constanty=1.0,
        n_repeatsx=nx,  # preferably use even number so that diagonals of reciprocal in mesh lattice lie along the exact diagonals, due to Monkhorst Pack grid.
        n_repeatsy=ny,
        k_nn=12.5,  # Nearest neighbour force constant
        k_nnn=0,  # Next nearest neighbour force constant
    )
    force_constants = build_force_constants_2d(system)
    np.testing.assert_allclose(force_constant_target, force_constants)
