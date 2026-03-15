from typing import Literal

import numpy as np
import pytest

import phonon_lifetime.cell.build as build_cell
from phonon_lifetime.cell import SuperCell
from phonon_lifetime.system import as_supercell, with_nearest_neighbor_forces
from phonon_lifetime.system._system import (  # noqa: PLC2701
    _get_offset_in_initial,
)


def _build_pristine_force_constant_matrix_slow(
    spring_constant: tuple[float, float, float],
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]:
    """Get the pristine force constant matrix.

    Mirrors ``with_nearest_neighbor_forces`` behavior, where forces are computed
    from the primitive cell to an expanded supercell with repeats ``3`` on
    periodic axes and ``1`` otherwise.

    """
    nx, ny, nz = n_repeats
    kx, ky, kz = spring_constant
    strain_repeats = (
        3 if kx != 0 else 1,
        3 if ky != 0 else 1,
        3 if kz != 0 else 1,
    )
    fx, fy, fz = (
        nx * strain_repeats[0],
        ny * strain_repeats[1],
        nz * strain_repeats[2],
    )

    def idx(ix: int, iy: int, iz: int) -> int:
        return np.ravel_multi_index((ix, iy, iz), (fx, fy, fz)).item()

    # Build force constants on the expanded grid, then keep primitive rows.
    n_primitive = nx * ny * nz
    n = fx * fy * fz
    fc = np.zeros((n, n, 3, 3), float)

    for ix in range(fx):
        for iy in range(fy):
            for iz in range(fz):
                i = idx(ix, iy, iz)

                if kx != 0:
                    jx_p = idx((ix + 1) % fx, iy, iz)
                    jx_m = idx((ix - 1) % fx, iy, iz)
                    fc[i, i, 0, 0] += 2 * kx
                    fc[i, jx_p, 0, 0] -= kx
                    fc[i, jx_m, 0, 0] -= kx

                if ky != 0:
                    jy_p = idx(ix, (iy + 1) % fy, iz)
                    jy_m = idx(ix, (iy - 1) % fy, iz)
                    fc[i, i, 1, 1] += 2 * ky
                    fc[i, jy_p, 1, 1] -= ky
                    fc[i, jy_m, 1, 1] -= ky

                if kz != 0:
                    jz_p = idx(ix, iy, (iz + 1) % fz)
                    jz_m = idx(ix, iy, (iz - 1) % fz)
                    fc[i, i, 2, 2] += 2 * kz
                    fc[i, jz_p, 2, 2] -= kz
                    fc[i, jz_m, 2, 2] -= kz

    return fc[:n_primitive]


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


def test_build_force_matrix_x() -> None:
    spring_constant = 1
    n_repeats = (4, 1, 1)
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    cell = SuperCell(cell, n_repeats=n_repeats)
    system = with_nearest_neighbor_forces(
        cell,
        spring_constant=spring_constant,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    actual = system.strain
    desired = _build_pristine_force_constant_matrix_slow(
        (spring_constant, 0, 0), n_repeats
    )
    np.testing.assert_array_equal(actual, desired)
    system = with_nearest_neighbor_forces(
        cell.primitive_cell,
        spring_constant=spring_constant,
        periodic=(True, False, False),
        cutoff=1.1,
    )
    supercell_system = as_supercell(system, n_repeats=n_repeats)
    actual = supercell_system.strain
    np.testing.assert_array_equal(actual, desired)


@pytest.mark.filterwarnings("ignore:Even n_repeats ")
def test_build_force_matrix_y() -> None:
    n_repeats = (1, 3, 1)
    spring_constant = 1
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    cell = SuperCell(cell, n_repeats=n_repeats)
    system = with_nearest_neighbor_forces(
        cell,
        spring_constant=spring_constant,
        periodic=(False, True, False),
        cutoff=1.1,
    )

    actual = system.strain
    desired = _build_pristine_force_constant_matrix_slow(
        (0, spring_constant, 0), n_repeats
    )
    np.testing.assert_array_equal(actual, desired)
    system = with_nearest_neighbor_forces(
        cell.primitive_cell,
        spring_constant=spring_constant,
        periodic=(False, True, False),
        cutoff=1.1,
    )
    supercell_system = as_supercell(system, n_repeats=n_repeats)
    actual = supercell_system.strain
    np.testing.assert_array_equal(actual, desired)


def test_build_force_matrix_explicit() -> None:
    n_repeats = (3, 1, 1)
    spring_constant = 1
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    cell = SuperCell(cell, n_repeats=n_repeats)
    system = with_nearest_neighbor_forces(
        cell,
        spring_constant=spring_constant,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    actual = system.strain
    np.testing.assert_array_equal(
        actual[:, :3, 0, 0], np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    )
    np.testing.assert_array_equal(actual[:, :, 1, 1], 0)
    np.testing.assert_array_equal(actual[:, :, 2, 2], 0)


def test_get_offset_in_initial() -> None:
    rng = np.random.default_rng()
    for n in range(1, 10):
        offsets = np.arange(-20, 20)
        valid_offsets = np.fft.fftfreq(n, d=1 / n)  # cspell: disable-line

        for offset in offsets:
            initial = tuple(rng.integers(-100, 100, size=3))
            final = (initial[0] + offset, initial[1], initial[2])
            actual = _get_offset_in_initial(initial, final, (n, 1, 1), (50, 1, 1))
            if offset in valid_offsets:
                expected = offset % n
                assert actual == expected, (
                    f"Expected {expected} but got {actual} for offset {offset} and n {n}"
                )
            else:
                assert actual is None, (
                    f"Expected None but got {actual} for offset {offset} and n {n}"
                )
