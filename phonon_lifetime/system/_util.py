from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from phonon_lifetime import System


def get_pristine_force_matrix(
    system: System,
) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]:
    """Get the pristine force constant matrix."""
    nx, ny, nz = system.n_repeats
    num_atoms = np.prod(system.n_repeats)

    # 1. Define the 3x3 stiffness tensors for each direction
    # If your spring_constant is (kx, ky, kz), these are diagonal
    kx, ky, kz = system.spring_constant
    phi_x = np.diag([kx, 0.0, 0.0])  # Only X-displacements cause X-forces
    phi_y = np.diag([0.0, ky, 0.0])
    phi_z = np.diag([0.0, 0.0, kz])

    # Initialize FC matrix: (N_atoms, N_atoms, 3, 3)
    fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=np.float64)

    # Create grid indices and flatten for mapping
    indices = np.arange(num_atoms).reshape((nx, ny, nz))
    target_atoms = np.arange(num_atoms)

    # 2. Fill Neighbor Interactions (Off-Diagonal)
    # We use np.roll to find neighbor indices across periodic boundaries
    for axis, phi in enumerate([phi_x, phi_y, phi_z]):
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


def get_supercell_cell(
    system: System,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the supercell lattice vectors.

    supercell_cell[i] is the vector (x, y, z) for the i'th lattice vector of the supercell.

    """
    return np.einsum("i,ij->ij", system.n_repeats, system.primitive_cell)


def get_scaled_positions(
    system: System,
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
    """Get the scaled positions of the atoms in the system.

    This gives the fraction along each of the ith lattice vector of the supercell.
    """
    # The index for all sites
    g = np.indices(system.n_repeats).reshape(3, -1)
    return (g / np.array(system.n_repeats)[:, None]).T


def get_atom_centres(
    system: System,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the centres of the atoms in the system."""
    # get_scaled_positions returns the positions along each axis j
    # for each atom i.
    # system.supercell_cell[j] is the cartesian vector for the j'th lattice vector of the supercell.
    return np.einsum(
        "ij,jk->ik", get_scaled_positions(system), get_supercell_cell(system)
    )
