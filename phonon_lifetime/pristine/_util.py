from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.neighborlist import neighbor_list

from phonon_lifetime.system._util import as_ase_atoms

from ._pristine import PristineSystem

if TYPE_CHECKING:
    from ase import Atoms

    from phonon_lifetime import System


def stiffness_from_spring_constant(
    spring_constant: tuple[float, float, float],
) -> np.ndarray[tuple[Literal[3], Literal[3], Literal[3]], np.dtype[np.float64]]:
    """Get the force constant matrix for a 1D chain with the given spring constant."""
    kx, ky, kz = spring_constant
    return np.array(
        [
            np.diag([kx, 0.0, 0.0]),  # X-direction
            np.diag([0.0, ky, 0.0]),  # Y-direction
            np.diag([0.0, 0.0, kz]),  # Z-direction
        ]
    )


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


def from_ase_atoms(atoms: Atoms, n_repeats: tuple[int, int, int]) -> PristineSystem:
    primitive_cell = atoms.get_cell()
    primitive_cell[2, 2] = 1
    return PristineSystem(
        mass=atoms.get_masses()[0],
        primitive_cell=primitive_cell,
        n_repeats=n_repeats,
        primitive_atom_fractions=atoms.get_scaled_positions(),
    )


def with_nearest_neighbour_force(
    system: System,
    spring_constant: float,
    *,
    cutoff: float = 2.460,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> PristineSystem:
    """Return a new PristineSystem with nearest neighbour forces added.

    The forces are added in the form of a spring force between nearest neighbours, with the given spring constant.
    The cutoff is used to determine which atoms are considered nearest neighbours.

    """
    as_pristine = system.as_pristine()
    as_ase = as_ase_atoms(as_pristine)
    as_ase.set_pbc(periodic)
    forces = np.zeros_like(system.forces)
    locations_i, locations_j, directions = neighbor_list("ijD", as_ase, cutoff=cutoff)
    for i, j, d in zip(locations_i, locations_j, directions, strict=False):
        direction = d / np.linalg.norm(d)
        np.testing.assert_allclose(1, np.linalg.norm(direction))
        forces[i, j] -= spring_constant * np.outer(direction, direction)
    for i in range(forces.shape[0]):
        forces[i, i, :, :] -= np.sum(forces[i, :, :, :], axis=0)
    return as_pristine.with_forces(forces=forces)
