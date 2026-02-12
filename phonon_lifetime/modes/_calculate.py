from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_lifetime.modes._modes import (
    PristineNormalModeResult,
    VacancyNormalModeResult,
)
from phonon_lifetime.system import get_scaled_positions

if TYPE_CHECKING:
    from phonon_lifetime.modes._modes import NormalModeResult
    from phonon_lifetime.system import System


def _get_normal_modes_vacancy(
    system: System,
    vacancy: list[int],
) -> VacancyNormalModeResult:
    n_vacancy = system.n_atoms - len(vacancy)
    all_positions = get_scaled_positions(system)

    cell = PhonopyAtoms(
        symbols=[system.element] * n_vacancy,
        cell=system.supercell_cell,
        scaled_positions=np.delete(all_positions, vacancy, axis=0),
    )

    phonon = Phonopy(
        unitcell=cell, supercell_matrix=np.eye(3), primitive_matrix=np.eye(3)
    )

    pristine_force_constants = _build_pristine_force_constant_matrix(system)
    phonon.force_constants = np.delete(
        np.delete(pristine_force_constants, vacancy, axis=0), vacancy, axis=1
    )

    phonon.run_mesh((1, 1, 1), with_eigenvectors=True, is_mesh_symmetry=False)

    mesh_dict = phonon.get_mesh_dict()

    return VacancyNormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"][0] * 1e12 * 2 * np.pi,
        modes=mesh_dict["eigenvectors"][0],
        vacancy=vacancy,
    )


def _build_pristine_force_constant_matrix(
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


def _get_normal_modes_pristine(
    system: System,
) -> PristineNormalModeResult:
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=system.primitive_cell,
        scaled_positions=[[0.0, 0.0, 0.0]],
    )

    phonon = Phonopy(
        unitcell=cell,
        supercell_matrix=np.diag(system.n_repeats),
        primitive_matrix=np.eye(3),
    )

    phonon.force_constants = _build_pristine_force_constant_matrix(system)

    phonon.run_mesh(system.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)

    mesh_dict = phonon.get_mesh_dict()

    return PristineNormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"] * 1e12 * 2 * np.pi,
        modes=mesh_dict["eigenvectors"],
        q_vals=mesh_dict["qpoints"],  # cspell: disable-line
    )


@overload
def calculate_normal_modes(
    system: System, *, vacancy: None = None
) -> PristineNormalModeResult: ...


@overload
def calculate_normal_modes(
    system: System, *, vacancy: list[int]
) -> VacancyNormalModeResult: ...


def calculate_normal_modes(
    system: System, *, vacancy: list[int] | None = None
) -> NormalModeResult:
    if vacancy is not None:
        return _get_normal_modes_vacancy(system, vacancy)
    return _get_normal_modes_pristine(system)
