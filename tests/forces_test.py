from typing import Literal

import numpy as np
import pytest

from phonon_lifetime import pristine
from phonon_lifetime.pristine._pristine import _recover_full_forces  # noqa: PLC2701
from phonon_lifetime.system import build


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


def pristine_forces_from_stiffness_tensor_square(
    stiffness: np.ndarray[
        tuple[Literal[3], Literal[3], Literal[3]], np.dtype[np.float64]
    ],
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
    """Get the pristine force constant matrix."""
    nx, ny, nz = n_repeats
    num_atoms = np.prod(n_repeats)

    # Initialize row for atom 0: (num_atoms, 3, 3)
    row_fc = np.zeros((1, num_atoms, 3, 3), dtype=np.float64)
    indices = np.arange(num_atoms).reshape((nx, ny, nz))

    # Find neighbor indices specifically for the atom at (0,0,0)
    # This is equivalent to seeing where '0' moved to after a roll
    for axis, phi in enumerate(stiffness):
        # Neighbor in positive direction
        idx_p = np.roll(indices, shift=-1, axis=axis)[0, 0, 0]
        row_fc[0, idx_p] -= phi

        # Neighbor in negative direction
        idx_m = np.roll(indices, shift=1, axis=axis)[0, 0, 0]
        row_fc[0, idx_m] -= phi

    # Acoustic Sum Rule: Self-interaction is the negative sum of all others
    row_fc[0, 0] -= np.sum(row_fc[0], axis=0)

    return row_fc


def full_forces_from_stiffness_tensor_square(
    stiffness: np.ndarray[
        tuple[Literal[3], Literal[3], Literal[3]], np.dtype[np.float64]
    ],
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
    """Get the pristine force constant matrix."""
    nx, ny, nz = n_repeats
    num_atoms = np.prod(n_repeats)

    # Initialize FC matrix: (N_atoms, N_atoms, 3, 3)
    fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=np.float64)

    # Create grid indices and flatten for mapping
    indices = np.arange(num_atoms).reshape((nx, ny, nz))
    target_atoms = np.arange(num_atoms)

    # 2. Fill Neighbor Interactions (Off-Diagonal)
    # We use np.roll to find neighbor indices across periodic boundaries
    for axis, phi in enumerate(stiffness):
        # Positive direction neighbor (+1)
        neighbors_p = np.roll(indices, shift=-1, axis=axis).ravel()
        fc[target_atoms, neighbors_p, :, :] -= phi

        # Negative direction neighbor (-1)
        neighbors_m = np.roll(indices, shift=1, axis=axis).ravel()
        fc[target_atoms, neighbors_m, :, :] -= phi

    # 3. Fill Self-Interactions (On-Diagonal)
    # The Acoustic Sum Rule requires Phi_ii = -sum(Phi_ij)
    # This ensures frequencies are zero at the Gamma point.
    for i in range(num_atoms):
        fc[i, i, :, :] -= np.sum(fc[i, :, :, :], axis=0)

    return fc


def test_recover_full_forces() -> None:
    rng = np.random.default_rng()
    stiffness_tensor = rng.random(size=(3, 3))
    pristine = pristine_forces_from_stiffness_tensor_square(stiffness_tensor, (2, 2, 2))
    full = full_forces_from_stiffness_tensor_square(stiffness_tensor, (2, 2, 2))
    restored = _recover_full_forces(pristine, (2, 2, 2))
    np.testing.assert_allclose(restored, full)


def test_build_force_matrix_x() -> None:
    spring_constant = 1
    n_repeats = (37, 1, 1)
    system = build.cubic(mass=10, distance=1.0, n_repeats=n_repeats, structure="simple")
    system = pristine.with_nearest_neighbor_forces(
        system,
        spring_constant=spring_constant,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    actual = system.forces
    desired = _build_pristine_force_constant_matrix_slow(
        (spring_constant, 0, 0), n_repeats
    )
    np.testing.assert_array_equal(actual, desired)
    actual = system.pristine_forces
    np.testing.assert_array_equal(actual, desired[0].reshape(1, -1, 3, 3))


@pytest.mark.filterwarnings("ignore:Even n_repeats ")
def test_build_force_matrix_y() -> None:
    n_repeats = (1, 2, 1)
    spring_constant = 1
    system = build.cubic(mass=10, distance=1.0, n_repeats=n_repeats, structure="simple")
    system = pristine.with_nearest_neighbor_forces(
        system,
        spring_constant=spring_constant,
        periodic=(False, True, False),
        cutoff=1.1,
    )

    actual = system.forces
    desired = _build_pristine_force_constant_matrix_slow(
        (0, spring_constant, 0), n_repeats
    )
    np.testing.assert_array_equal(actual, desired)
    actual = system.pristine_forces
    np.testing.assert_array_equal(actual, desired[0].reshape(1, -1, 3, 3))


def test_build_force_matrix_explicit() -> None:
    n_repeats = (3, 1, 1)
    spring_constant = 1
    system = build.cubic(mass=10, distance=1.0, n_repeats=n_repeats, structure="simple")
    system = pristine.with_nearest_neighbor_forces(
        system,
        spring_constant=spring_constant,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    actual = system.forces

    np.testing.assert_array_equal(
        actual[:, :, 0, 0], np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    )
    np.testing.assert_array_equal(actual[:, :, 1, 1], 0)
    np.testing.assert_array_equal(actual[:, :, 2, 2], 0)
