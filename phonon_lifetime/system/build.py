from typing import TYPE_CHECKING, Literal, cast

import ase.build
from ase import Atoms

if TYPE_CHECKING:
    import numpy as np

    from phonon_lifetime.pristine import PristineSystem


def from_ase_atoms(atoms: Atoms, n_repeats: tuple[int, int, int]) -> PristineSystem:
    """Build a system from an ASE Atoms object."""
    from phonon_lifetime.pristine import PristineSystem  # noqa: PLC0415

    return PristineSystem(
        primitive_masses=atoms.get_masses(),
        primitive_cell=atoms.cell.array,
        n_repeats=n_repeats,
        primitive_atom_fractions=atoms.get_scaled_positions(),
        primitive_symbols=atoms.get_chemical_symbols(),
    )


def graphene(
    *,
    mass: float,
    n_repeats: tuple[int, int, Literal[1]],
    distance: float = 2.460,
) -> PristineSystem:
    """Build a graphene system."""
    atoms = cast("Atoms", ase.build.graphene(a=distance))
    atoms.set_masses([mass] * len(atoms))

    cell = atoms.cell.array
    cell[2, 2] = 1
    atoms.set_cell(cell)
    return from_ase_atoms(atoms, n_repeats=n_repeats)


type CubicStructure = Literal["simple", "bcc", "fcc"]


def _as_ase_structure(structure: CubicStructure) -> str:
    if structure == "simple":
        return "sc"
    if structure == "bcc":
        return "bcc"
    if structure == "fcc":
        return "fcc"
    msg = f"Unknown structure: {structure}"
    raise ValueError(msg)


def cubic(
    *,
    mass: float,
    n_repeats: tuple[int, int, int],
    structure: CubicStructure,
    distance: float = 1.0,
) -> PristineSystem:
    """Build a simple cubic system."""
    cell = ase.build.bulk(
        name="C", crystalstructure=_as_ase_structure(structure), a=distance
    )
    cell.set_masses([mass] * len(cell))
    return from_ase_atoms(cell, n_repeats=n_repeats)


def from_primitive(
    *,
    mass: float,
    primitive_cell: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]],
    n_repeats: tuple[int, int, int],
) -> PristineSystem:
    """Build a system with a primitive cell."""
    cell = Atoms(
        positions=[(0.0, 0.0, 0.0)],
        cell=primitive_cell,
        pbc=True,
    )
    cell.set_masses([mass])
    return from_ase_atoms(cell, n_repeats=n_repeats)
