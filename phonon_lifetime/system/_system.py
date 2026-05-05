import warnings
from typing import Literal, cast

import numpy as np

from phonon_lifetime.cell import SuperCell, UnitCell


def _wrap_index(i: int, n: int) -> int:
    """Wrap the index i into the first bz."""
    # This maps i to the range [-n//2, (n-1)//2]
    return (i + n // 2) % n - n // 2


def _get_offset_in_initial(
    c_i: tuple[np.int64, ...] | tuple[int, ...],
    c_j: tuple[np.int64, ...] | tuple[int, ...],
    n_repeats_initial: tuple[int, int, int],
    n_repeats_final: tuple[int, int, int],
) -> int | None:
    """Get the offset of j relative to i in the initial system."""
    out = []
    for i in range(3):
        offset = int(c_j[i] - c_i[i])
        # Wrap the offset to be within the bounds of the initial system, which is centered around the origin.
        offset = _wrap_index(offset, n_repeats_final[i])
        min_rep = -n_repeats_initial[i] / 2
        max_rep = n_repeats_initial[i] / 2
        if offset < min_rep or offset >= max_rep:
            return None
        out.append(offset % n_repeats_initial[i])
    return np.ravel_multi_index(tuple(out), n_repeats_initial)


class StrainSystem[C: UnitCell = UnitCell]:
    """Represents a system for phonon calculations."""

    def __init__(
        self,
        cell: C,
        strain: np.ndarray[
            tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]
        ],
        strain_repeats: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self._cell = cell
        self._strain = strain
        self._strain_repeats = strain_repeats
        self.__post_init__()

    def __post_init__(self) -> None:

        if self._cell.n_atoms != self._strain.shape[0]:
            msg = f"Number of atoms in the cell should match the second dimension of the strain, but got {self._cell.n_atoms} atoms and strain with shape {self._strain.shape}."
            raise ValueError(msg)
        if np.prod(self._strain_repeats) * self._cell.n_atoms != self._strain.shape[1]:
            msg = f"Product of strain_repeats should match the first dimension of the strain, but got strain_repeats={self._strain_repeats} and strain with shape {self._strain.shape}."
            raise ValueError(msg)

    @property
    def cell(self) -> C:
        """Get the cell of the system."""
        return self._cell

    @property
    def strain(
        self,
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
        """Get the strain of the system."""
        return self._strain

    @property
    def strain_repeats(self) -> tuple[int, int, int]:
        """Get the strain repeats of the system."""
        return self._strain_repeats


def _get_repeated_strain(
    strian_system: StrainSystem,
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
    """Get strain for a repeated cell while preserving strain_repeats semantics."""
    if any(
        i <= 1 and j != 1
        for i, j in zip(strian_system.strain_repeats, n_repeats, strict=True)
    ):
        msg = (
            "strain_repeats should be greater than 1 for correct strain if n_repeats is greater than 1, "
            f"but got strain_repeats={strian_system.strain_repeats} and n_repeats={n_repeats}. "
            "This is due to a limitation in the current implementation of the code, and may lead to incorrect strain being used in the phonon calculations."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)

    n_repeats_with_strain = cast(
        "tuple[int, int, int]",
        tuple(
            i * j for i, j in zip(n_repeats, strian_system.strain_repeats, strict=True)
        ),
    )
    n_base_atoms = strian_system.cell.n_atoms
    n_new_repeats = np.prod(n_repeats).item()
    n_strain_repeats = np.prod(strian_system.strain_repeats).item()
    out = np.zeros(
        (
            n_base_atoms * n_new_repeats,
            n_base_atoms * n_new_repeats * n_strain_repeats,
            3,
            3,
        ),
        dtype=np.float64,
    )

    for i in range(out.shape[0]):
        i_base_atom = i % n_base_atoms
        i_repeat = np.unravel_index(i // n_base_atoms, n_repeats)
        for j in range(out.shape[1]):
            j_base_atom = j % n_base_atoms
            j_repeat_idx = (j // n_base_atoms) % n_new_repeats
            j_strain_idx = (j // n_base_atoms) // n_new_repeats

            j_repeat = np.unravel_index(j_repeat_idx, n_repeats)
            j_strain = np.unravel_index(j_strain_idx, strian_system.strain_repeats)
            j_combined = tuple(
                jr + nr * js
                for jr, nr, js in zip(j_repeat, n_repeats, j_strain, strict=True)
            )

            j_relative = _get_offset_in_initial(
                i_repeat,
                j_combined,
                strian_system.strain_repeats,
                n_repeats_with_strain,
            )
            if j_relative is None:
                continue

            out[i, j] = strian_system.strain[
                i_base_atom,
                j_relative * n_base_atoms + j_base_atom,
            ]

    return out  # ty:ignore[invalid-return-type]


def as_supercell[C: UnitCell](
    strian_system: StrainSystem[C],
    n_repeats: tuple[int, int, int],
) -> StrainSystem[SuperCell[C]]:
    """Get the strain of the supercell."""
    return StrainSystem(
        cell=SuperCell(
            primitive_cell=strian_system.cell,
            n_repeats=n_repeats,
        ),
        strain=_get_repeated_strain(strian_system, n_repeats=n_repeats),
        # Keep the original strain basis; supercell conversion changes the cell,
        # not the interaction repeat metadata carried by the force constants.
        strain_repeats=strian_system.strain_repeats,
    )
