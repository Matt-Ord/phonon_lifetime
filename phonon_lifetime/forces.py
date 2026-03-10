import dataclasses
from typing import Literal

import numpy as np


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
    forces: np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]],
    n_repeats_initial: tuple[int, int, int],
    n_repeats_final: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
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


def _recover_pristine_forces(
    forces: np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]],
    n_repeats_initial: tuple[int, int, int],
    n_repeats_final: tuple[int, int, int],
) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
    """Recover the full forces from the pristine forces."""
    n_primitive = forces.shape[0]
    n_final_atoms = n_primitive * np.prod(n_repeats_final).item()
    full_forces = np.zeros((n_primitive, n_final_atoms, 3, 3), dtype=np.float64)
    for i_in_primitive in range(n_primitive):
        c_i = (0, 0, 0)
        for j in range(n_final_atoms):
            p_j = j % n_primitive
            c_j = np.unravel_index(j // n_primitive, n_repeats_final)
            cj_relative = _get_offset_in_initial(
                c_i, c_j, n_repeats_initial, n_repeats_final
            )

            if cj_relative is None:
                continue

            j_relative_to_i = cj_relative * n_primitive + p_j
            full_forces[i_in_primitive, j] = forces[i_in_primitive, j_relative_to_i]
    return full_forces  # ty:ignore[invalid-return-type]


@dataclasses.dataclass(kw_only=True, frozen=True)
class PristineStrainTensor:
    """Represents the strain tensor for a system of atoms."""

    data: np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]
    n_repeats: tuple[int, int, int]

    @property
    def n_primitive_atoms(self) -> int:
        """Number of atoms in the primitive cell."""
        return self.data.shape[0]

    def calculate_full_tensor(
        self,
        n_repeats: tuple[int, int, int],
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        """Return the full strain tensor for a system with shape (n_repeats)."""
        return _recover_full_forces(self.data, self.n_repeats, n_repeats)

    def calculate_pristine_tensor(
        self,
        n_repeats: tuple[int, int, int],
    ) -> np.ndarray[tuple[int, int, Literal[3], Literal[3]], np.dtype[np.float64]]:
        """Return the pristine strain tensor for a system with shape (n_repeats)."""
        return _recover_pristine_forces(self.data, self.n_repeats, n_repeats)


def zero_strain_tensor(n_primitive_atoms: int) -> PristineStrainTensor:
    """Return a zero strain tensor."""
    return PristineStrainTensor(
        data=np.zeros((n_primitive_atoms, 0, 3, 3), dtype=np.float64),  # ty:ignore[invalid-argument-type]
        n_repeats=(0, 0, 0),
    )
