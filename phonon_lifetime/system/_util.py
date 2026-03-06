from typing import TYPE_CHECKING, Literal

from ase import Atoms
from ase.cell import Cell

if TYPE_CHECKING:
    import numpy as np

    from phonon_lifetime import System
    from phonon_lifetime.pristine import PristineSystem


def as_primitive(system: System) -> PristineSystem:

    from phonon_lifetime.pristine._pristine import PristineSystem  # noqa: PLC0415

    return PristineSystem(
        primitive_masses=system.as_pristine().primitive_masses,
        primitive_cell=system.primitive_cell,
        n_repeats=(1, 1, 1),
        primitive_atom_fractions=system.primitive_atom_fractions,
        primitive_symbols=system.as_pristine().primitive_symbols,
    )


def as_ase_atoms(system: PristineSystem) -> Atoms:
    return Atoms(
        symbols=system.primitive_symbols,
        masses=system.primitive_masses,
        cell=Cell(system.primitive_cell),
        scaled_positions=system.primitive_atom_fractions,
    ).repeat(system.n_repeats)


def get_supercell_cell(
    system: System,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the supercell lattice vectors.

    supercell_cell[i] is the vector (x, y, z) for the i'th lattice vector of the supercell.

    """
    return as_ase_atoms(system.as_pristine()).get_cell()


def get_atom_fractions(
    system: System,
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
    """Get the scaled positions of the atoms in the system.

    This gives the fraction along each of the ith lattice vector of the supercell.
    """
    primitive_atoms = as_ase_atoms(as_primitive(system))
    atoms = as_ase_atoms(system.as_pristine())
    return primitive_atoms.cell.scaled_positions(atoms.get_positions())


def get_atom_supercell_fractions(
    system: System,
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
    """Get the positions of the atoms in the system in cartesian coordinates."""
    return as_ase_atoms(system.as_pristine()).get_scaled_positions()


def get_atom_centres(
    system: System,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the centres of the atoms in the system."""
    return as_ase_atoms(system.as_pristine()).get_positions()
