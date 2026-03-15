import warnings
from typing import Literal

import numpy as np
from ase.filters import ExpCellFilter
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS  # cspell: disable-line
from ase.phonons import Phonons

from phonon_lifetime.cell import UnitCell, as_ase_atoms
from phonon_lifetime.system._system import StrainSystem

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="e3nn.o3._wigner"
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="mace.calculators.mace"
        )

        from mace.calculators import (  # ty:ignore[unresolved-import, unused-ignore-comment]
            mace_mp,
        )
except ImportError:
    pass


def with_nearest_neighbor_forces[C: UnitCell](
    cell: C,
    spring_constant: float,
    *,
    cutoff: float = 2.460,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> StrainSystem[C]:
    """Return a new PristineSystem with nearest neighbor forces added.

    The forces are added in the form of a spring force between nearest neighbor, with the given spring constant.
    The cutoff is used to determine which atoms are considered nearest neighbor.

    """
    n_repeats: tuple[int, int, int] = tuple(3 if p else 1 for p in periodic)  # ty:ignore[invalid-assignment]
    n_primitive_atoms = cell.n_atoms
    data = np.zeros(
        (n_primitive_atoms, np.prod(n_repeats) * n_primitive_atoms, 3, 3),
        dtype=np.float64,
    )

    as_ase = as_ase_atoms(cell).repeat(n_repeats)
    as_ase.set_pbc(periodic)

    locations_i, locations_j, directions = neighbor_list("ijD", as_ase, cutoff=cutoff)
    for i, j, d in zip(locations_i, locations_j, directions, strict=False):
        if i >= data.shape[0]:
            continue
        direction = d / np.linalg.norm(d)
        np.testing.assert_allclose(1, np.linalg.norm(direction))
        data[i, j] -= spring_constant * np.outer(direction, direction)
    for i in range(data.shape[0]):
        data[i, i, :, :] -= np.sum(data[i, :, :, :], axis=0)
    return StrainSystem(cell=cell, strain=data, strain_repeats=n_repeats)


def _phonopy_strain_from_ase(
    ase_forces: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    *,
    n_primitive_atoms: int,
    strain_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
    """Convert the ASE force constants to the phonopy format."""
    n_total_repeats = np.prod(strain_repeats).item()

    # The ASE forces are in the shape (N, (u i), (v j)), where
    # N: Number of unit cells in supercell
    # (u i), (v j) are (n_primitive_atoms * 3) length flattend dimensions.
    reshaped_ase_forces = ase_forces.reshape(
        n_total_repeats, n_primitive_atoms, 3, n_primitive_atoms, 3
    )

    # 2. Use einsum to rearrange
    # final shape (n_primitive_atoms, n_repeats , n_primitive_atoms, 3, 3)
    compact_fc = np.einsum(
        "nuivj -> unvij",  # cspell: disable-line
        reshaped_ase_forces,
    )
    # Phonopy convention: FC[unit_atom, super_atom, direction_i, direction_j]
    return compact_fc.reshape(
        n_primitive_atoms, n_total_repeats * n_primitive_atoms, 3, 3
    )


def with_ase_forces[C: UnitCell](
    cell: C,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    n_repeats: tuple[int, int, int] | None = None,
) -> StrainSystem[C]:
    """Return a new StrainSystem with forces calculated using ASE.

    Parameters
    ----------
    cell: C
        The system to calculate forces for. The system should be a pristine system, or at least have a well defined primitive cell.
    periodic: tuple[bool, bool, bool]
        Whether to apply periodic boundary conditions in each direction when calculating forces. This will affect which atoms are considered nearest neighbors.
    n_repeats: tuple[int, int, int] | None
        The number of repeats to use when calculating forces. If None, will simulate the full system.

    """
    ase_unitcell = as_ase_atoms(cell)
    ase_unitcell.set_pbc(periodic)
    calc = mace_mp(
        model="mh-1",
        head="omat_pbe",  # cspell: disable-line
        default_dtype="float64",
    )
    ase_unitcell.calc = calc

    # Relax the unit cell, so equilibrium forces are zero.
    ecf = ExpCellFilter(ase_unitcell)
    opt = BFGS(ecf)  # ty:ignore[invalid-argument-type] # cspell: disable-line
    opt.run(fmax=1e-4)  # cspell: disable-line

    # Calculate forces on the supercell
    strain_repeats = (
        tuple[int, int, int](3 if p else 1 for p in periodic)  # ty:ignore[invalid-argument-type]
        if n_repeats is None
        else n_repeats
    )
    ase_phonons = Phonons(ase_unitcell, calc, supercell=strain_repeats)
    ase_phonons.cache.clear()
    ase_phonons.run()
    ase_phonons.read()

    return StrainSystem[C](
        cell=cell,
        strain=_phonopy_strain_from_ase(
            ase_phonons.get_force_constant(),
            n_primitive_atoms=cell.n_atoms,
            strain_repeats=strain_repeats,
        ),
        strain_repeats=strain_repeats,
    )


def with_zero_forces[C: UnitCell](cell: C) -> StrainSystem[C]:
    """Return a new StrainSystem with zero forces."""
    n_repeats: tuple[int, int, int] = (1, 1, 1)
    n_primitive_atoms = cell.n_atoms
    data = np.zeros(
        (n_primitive_atoms, np.prod(n_repeats) * n_primitive_atoms, 3, 3),
        dtype=np.float64,
    )
    return StrainSystem(cell=cell, strain=data, strain_repeats=n_repeats)
