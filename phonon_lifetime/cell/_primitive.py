from typing import TYPE_CHECKING, override

import numpy as np

from ._cell import UnitCell

if TYPE_CHECKING:
    from ase import Atoms


class PrimitiveCell(UnitCell):
    """Represents the primitive cell of a defect free system."""

    _masses: np.ndarray[tuple[int], np.dtype[np.floating]]
    _symbols: list[str]
    _vectors: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    _atom_fractions: np.ndarray[tuple[int, int], np.dtype[np.floating]]

    def __init__(
        self,
        *,
        masses: np.ndarray[tuple[int], np.dtype[np.floating]],
        symbols: list[str] | None = None,
        vectors: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        atom_fractions: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> None:
        super().__init__()
        self._masses = masses
        self._symbols = (
            symbols if symbols is not None else ["C" for i in range(masses.size)]
        )
        self._vectors = vectors
        self._atom_fractions = atom_fractions
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.vectors.shape != (3, 3):
            msg = f"Primitive cell should have shape (3, 3), but got {self.vectors.shape}."
            raise ValueError(msg)

        if len(self.symbols) != self.n_atoms:
            msg = f"Number of symbols should match number of atoms, but got {len(self.symbols)} symbols and {self.n_atoms} atoms."
            raise ValueError(msg)

        if self.atom_fractions.shape[1] != 3:  # noqa: PLR2004
            msg = f"Atom fractions should have shape (n_atoms, 3), but got {self.atom_fractions.shape}."
            raise ValueError(msg)

        if self.atom_fractions.shape[0] != self.n_atoms:
            msg = f"Number of atom fractions should match number of atoms, but got {self.atom_fractions.shape[0]} atom fractions and {self.n_atoms} atoms."
            raise ValueError(msg)

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self._masses

    @property
    @override
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    @override
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._vectors

    @property
    @override
    def atom_fractions(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._atom_fractions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrimitiveCell):
            return NotImplemented
        return (
            np.allclose(self.masses, other.masses)
            and self.symbols == other.symbols
            and np.allclose(self.vectors, other.vectors)
            and np.allclose(self.atom_fractions, other.atom_fractions)
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.masses),
                tuple(self.symbols),
                tuple(map(tuple, self.vectors)),
                tuple(map(tuple, self.atom_fractions)),
            )
        )


def from_ase_atoms(atoms: Atoms) -> PrimitiveCell:
    """Convert an ASE Atoms object to a PrimitiveCell."""
    return PrimitiveCell(
        masses=atoms.get_masses(),
        symbols=atoms.get_chemical_symbols(),
        vectors=atoms.get_cell().array,
        atom_fractions=atoms.get_scaled_positions(),
    )
