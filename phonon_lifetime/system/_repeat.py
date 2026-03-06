import warnings
from typing import TYPE_CHECKING, Literal, override

import numpy as np

from phonon_lifetime.system._system import System

if TYPE_CHECKING:
    from phonon_lifetime.modes import NormalMode, NormalModes
    from phonon_lifetime.pristine import PristineSystem


class RepeatSystem(System):
    """A system that repeats another system."""

    def __init__(self, system: System, n_repeats: tuple[int, int, int]) -> None:
        self._system = system
        self._n_repeats = n_repeats

    @property
    @override
    def masses(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.tile(self._system.masses, np.prod(self._n_repeats).item()).ravel()

    @property
    @override
    def symbols(self) -> list[str]:
        n_supercell = np.prod(self._n_repeats).item()
        return [s for _ in range(n_supercell) for s in self._system.symbols]

    @property
    @override
    def primitive_cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._system.primitive_cell

    @property
    @override
    def forces(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        raise NotImplementedError

    @property
    @override
    def n_repeats(self) -> tuple[int, int, int]:
        return tuple(
            a * b for a, b in zip(self._system.n_repeats, self._n_repeats, strict=True)
        )

    @override
    def as_pristine(self) -> PristineSystem:

        from phonon_lifetime.pristine._pristine import PristineSystem  # noqa: PLC0415

        warnings.warn(
            "Converting a RepeatSystem to a PristineSystem will currently set all forces to zero. This should be fixed in a future change.",
            UserWarning,
            stacklevel=2,
        )
        return PristineSystem(
            primitive_masses=self._system.as_pristine().primitive_masses,
            primitive_symbols=self._system.as_pristine().primitive_symbols,
            primitive_cell=self.primitive_cell,
            n_repeats=self.n_repeats,
            primitive_atom_fractions=self._system.as_pristine().primitive_atom_fractions,
        )

    @override
    def get_modes(self) -> NormalModes[System]:
        from phonon_lifetime.modes._util import repeat_modes  # noqa: PLC0415

        return repeat_modes(self._system.get_modes(), n_repeats=self._n_repeats)

    @override
    def get_mode(self, idx: int) -> NormalMode[System]:

        from phonon_lifetime.modes._util import repeat_mode  # noqa: PLC0415

        return repeat_mode(self._system.get_mode(idx), n_repeats=self._n_repeats)

    @property
    def inner_system(self) -> System:
        """The original system that is being repeated."""
        return self._system

    @property
    def n_primitive_atoms(self) -> int:
        return self._system.n_primitive_atoms

    @property
    def primitive_atom_fractions(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return self._system.primitive_atom_fractions
