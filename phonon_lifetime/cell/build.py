from typing import TYPE_CHECKING, Literal, cast

import ase.build
from ase import Atoms

from ._primitive import PrimitiveCell, from_ase_atoms

if TYPE_CHECKING:
    import numpy as np


def graphene(
    *,
    mass: float,
    distance: float = 2.460,
) -> PrimitiveCell:
    """Build a graphene system."""
    atoms = cast("Atoms", ase.build.graphene(a=distance))
    atoms.set_masses([mass] * len(atoms))

    cell = atoms.cell.array
    cell[2, 2] = 1
    atoms.set_cell(cell)
    return from_ase_atoms(atoms)


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
    structure: CubicStructure,
    distance: float = 1.0,
) -> PrimitiveCell:
    """Build a simple cubic system."""
    cell = ase.build.bulk(
        name="C", crystalstructure=_as_ase_structure(structure), a=distance
    )
    cell.set_masses([mass] * len(cell))
    return from_ase_atoms(cell)


def from_primitive(
    *,
    mass: float,
    primitive_cell: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]],
) -> PrimitiveCell:
    """Build a system with a primitive cell."""
    cell = Atoms(
        positions=[(0.0, 0.0, 0.0)],
        cell=primitive_cell,
        pbc=True,
    )
    cell.set_masses([mass])
    return from_ase_atoms(cell)
