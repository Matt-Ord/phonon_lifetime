import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.filters import ExpCellFilter
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS  # cspell: disable-line
from ase.phonons import Phonons

from phonon_lifetime.system import as_primitive
from phonon_lifetime.system._util import as_ase_atoms
from phonon_lifetime.system.build import from_ase_atoms

if TYPE_CHECKING:
    from phonon_lifetime import System

    from ._pristine import PristineSystem

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


def with_nearest_neighbor_forces(
    system: System,
    spring_constant: float,
    *,
    cutoff: float = 2.460,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> PristineSystem:
    """Return a new PristineSystem with nearest neighbor forces added.

    The forces are added in the form of a spring force between nearest neighbor, with the given spring constant.
    The cutoff is used to determine which atoms are considered nearest neighbor.

    """
    as_pristine = system.as_pristine()
    as_ase = as_ase_atoms(as_pristine)
    as_ase.set_pbc(periodic)
    forces = np.zeros_like(as_pristine.pristine_forces)

    locations_i, locations_j, directions = neighbor_list("ijD", as_ase, cutoff=cutoff)
    for i, j, d in zip(locations_i, locations_j, directions, strict=False):
        if i >= forces.shape[0]:
            continue
        direction = d / np.linalg.norm(d)
        np.testing.assert_allclose(1, np.linalg.norm(direction))
        forces[i, j] -= spring_constant * np.outer(direction, direction)
    for i in range(forces.shape[0]):
        forces[i, i, :, :] -= np.sum(forces[i, :, :, :], axis=0)
    return as_pristine.with_forces(forces=forces)


def _phonopy_forces_from_ase(
    ase_forces: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    *,
    n_primitive_atoms: int,
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
    n_repeats = ase_forces.shape[0]

    # The ASE forces are in the shape (N, (u i), (v j)), where
    # N: Number of unit cells in supercell
    # (u i), (v j) are (n_primitive_atoms * 3) length flattend dimensions.
    reshaped_ase_forces = ase_forces.reshape(
        n_repeats, n_primitive_atoms, 3, n_primitive_atoms, 3
    )

    # 2. Use einsum to rearrange
    # final shape (n_primitive_atoms, n_repeats , n_primitive_atoms, 3, 3)
    compact_fc = np.einsum(
        "nuivj -> unvij",  # cspell: disable-line
        reshaped_ase_forces,
    )
    # Phonopy convention: FC[unit_atom, super_atom, direction_i, direction_j]
    return compact_fc.reshape(n_primitive_atoms, n_repeats * n_primitive_atoms, 3, 3)


def with_ase_forces(
    system: System,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> PristineSystem:
    """Return a new PristineSystem with nearest neighbor forces added.

    The forces are added in the form of a spring force between nearest neighbor, with the given spring constant.
    The cutoff is used to determine which atoms are considered nearest neighbor.

    """
    pristine = system.as_pristine()
    ase_unitcell = as_ase_atoms(as_primitive(pristine))
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
    opt.run(fmax=0.01)  # cspell: disable-line

    # Calculate forces on the supercell
    ase_phonons = Phonons(ase_unitcell, calc, supercell=system.n_repeats)
    ase_phonons.cache.clear()
    ase_phonons.run()
    ase_phonons.read()

    return from_ase_atoms(ase_unitcell, n_repeats=system.n_repeats).with_forces(
        forces=_phonopy_forces_from_ase(
            ase_phonons.get_force_constant(),
            n_primitive_atoms=system.n_primitive_atoms,
        )
    )
