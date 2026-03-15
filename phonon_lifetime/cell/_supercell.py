from typing import TYPE_CHECKING, override

from phonon_lifetime.cell import UnitCell, as_ase_atoms

if TYPE_CHECKING:
    import numpy as np


class SuperCell[C: UnitCell](UnitCell):
    """Represents the supercell of a system."""

    def __init__(
        self,
        primitive_cell: C,
        n_repeats: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self._primitive_cell = primitive_cell
        self._n_repeats = n_repeats

    @property
    def primitive_cell(self) -> C:
        """Get the primitive cell of the system."""
        return self._primitive_cell

    @property
    def n_repeats(self) -> tuple[int, int, int]:
        """Get the number of repeats of the primitive cell in each direction."""
        return self._n_repeats

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return as_ase_atoms(self._primitive_cell).repeat(self._n_repeats).get_masses()

    @property
    @override
    def symbols(self) -> list[str]:
        return (
            as_ase_atoms(self._primitive_cell)
            .repeat(self._n_repeats)
            .get_chemical_symbols()
        )

    @property
    @override
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return (
            as_ase_atoms(self._primitive_cell).repeat(self._n_repeats).get_cell().array
        )

    @property
    @override
    def atom_fractions(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return (
            as_ase_atoms(self._primitive_cell)
            .repeat(self._n_repeats)
            .get_scaled_positions()
        )
