import numpy as np
import pytest

from phonon_lifetime.pristine import PristineSystem


def _build_pristine_force_constant_matrix_slow(
    spring_constant: tuple[float, float, float],
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]:
    """Get the pristine force constant matrix."""
    nx, ny, nz = n_repeats
    kx, ky, kz = spring_constant

    def idx(ix: int, iy: int, iz: int) -> int:
        return np.ravel_multi_index((ix, iy, iz), n_repeats).item()

    # 1) pristine fc on full grid
    n = nx * ny * nz
    fc = np.zeros((n, n, 3, 3), float)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                i = idx(ix, iy, iz)

                jx_p = idx((ix + 1) % nx, iy, iz)
                jx_m = idx((ix - 1) % nx, iy, iz)
                fc[i, i, 0, 0] += 2 * kx
                fc[i, jx_p, 0, 0] -= kx
                fc[i, jx_m, 0, 0] -= kx

                jy_p = idx(ix, (iy + 1) % ny, iz)
                jy_m = idx(ix, (iy - 1) % ny, iz)
                fc[i, i, 1, 1] += 2 * ky
                fc[i, jy_p, 1, 1] -= ky
                fc[i, jy_m, 1, 1] -= ky

                jz_p = idx(ix, iy, (iz + 1) % nz)
                jz_m = idx(ix, iy, (iz - 1) % nz)
                fc[i, i, 2, 2] += 2 * kz
                fc[i, jz_p, 2, 2] -= kz
                fc[i, jz_m, 2, 2] -= kz

    return fc


def test_build_force_matrix_x() -> None:
    spring_constant = (1, 0, 0)
    n_repeats = (37, 1, 1)
    system = PristineSystem.from_spring_constant(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=n_repeats,
        spring_constant=spring_constant,
    )

    actual = system.forces
    desired = _build_pristine_force_constant_matrix_slow(spring_constant, n_repeats)
    np.testing.assert_array_equal(actual, desired)
    actual = system.pristine_forces
    np.testing.assert_array_equal(actual, desired[0])


@pytest.mark.filterwarnings("ignore:Even n_repeats ")
def test_build_force_matrix_y() -> None:
    n_repeats = (37, 2, 1)
    spring_constant = (0, 1, 0)
    system = PristineSystem.from_spring_constant(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=n_repeats,
        spring_constant=spring_constant,
    )

    actual = system.forces
    desired = _build_pristine_force_constant_matrix_slow(spring_constant, n_repeats)
    np.testing.assert_array_equal(actual, desired)
    actual = system.pristine_forces
    np.testing.assert_array_equal(actual, desired[0])


def test_build_force_matrix_y_flat() -> None:
    n_repeats = (37, 1, 1)
    spring_constant = (0, 1, 0)

    system = PristineSystem.from_spring_constant(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=n_repeats,
        spring_constant=spring_constant,
    )

    actual = system.forces
    desired = _build_pristine_force_constant_matrix_slow(spring_constant, n_repeats)
    np.testing.assert_array_equal(actual, desired)
    actual = system.pristine_forces
    np.testing.assert_array_equal(actual, desired[0])


def test_build_force_matrix_explicit() -> None:
    system = PristineSystem.from_spring_constant(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(3, 1, 1),
        spring_constant=(1, 0, 0),
    )

    actual = system.forces

    np.testing.assert_array_equal(
        actual[:, :, 0, 0], np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    )
    np.testing.assert_array_equal(actual[:, :, 1, 1], 0)
    np.testing.assert_array_equal(actual[:, :, 2, 2], 0)
