import warnings
from typing import Literal

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


def _recover_full_forces(
    forces: np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]],
    n_repeats_initial: tuple[int, int, int],
    n_repeats_final: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
    """Recover the full forces from the pristine forces."""
    n_primitive = forces.shape[0]
    n_final_atoms = n_primitive * np.prod(n_repeats_final).item()
    full_forces = np.zeros((n_final_atoms, n_final_atoms, 3, 3), dtype=np.float64)
    for i in range(n_final_atoms):
        i_in_primitive = i % n_primitive
        c_i = np.unravel_index(i // n_primitive, n_repeats_final)
        for j in range(n_final_atoms):
            p_j = j % n_primitive
            c_j = np.unravel_index(j // n_primitive, n_repeats_final)
            cj_relative = _get_offset_in_initial(
                c_i, c_j, n_repeats_initial, n_repeats_final
            )

            if cj_relative is None:
                continue

            j_relative_to_i = cj_relative * n_primitive + p_j
            full_forces[i, j] = forces[i_in_primitive, j_relative_to_i]
    return full_forces  # ty:ignore[invalid-return-type]


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


def _get_full_strain(
    strian_system: StrainSystem,
    n_repeats: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.floating]]:
    """Get the full strain of the system."""
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
    return _recover_full_forces(
        strian_system.strain,
        n_repeats_initial=strian_system.strain_repeats,
        n_repeats_final=n_repeats,
    )


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
        strain=_get_full_strain(strian_system, n_repeats=n_repeats),
        strain_repeats=(1, 1, 1),
    )
