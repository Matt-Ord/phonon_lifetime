from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms  # type: ignore[import]

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(kw_only=True, frozen=True)
class System:
    """Represents a lattice system used for phonon calculations."""

    element: str
    cell: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    n_repeats: tuple[int, int, int]
    spring_constant: tuple[float, float, float]

    @property
    def mass(self) -> float:
        """Mass of the element in atomic mass units."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=self.cell,
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


@dataclass(kw_only=True, frozen=True)
class NormalModeResult:
    """Result of a normal mode calculation for a phonon system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]
    """The normal mode frequencies in angular frequency units."""
    modes: np.ndarray[Any, np.dtype[np.floating]]
    """The eigenvectors (normal modes) of the system."""
    q_vals: np.ndarray[Any, np.dtype[np.floating]]
    """The reduced wave vectors for the normal modes."""

    def to_human_readable(self) -> str:
        """Convert the result to a text representation."""
        return (
            f"Calculating normal modes for system: {self.system}\n"
            "Normal mode frequencies (omega):\n"
            f"{np.array2string(self.omega, precision=6, separator=', ')}\n"
            "Wave vectors (q):\n"
            f"{np.array2string(self.q_vals, precision=6, separator=', ')}\n"
            "Normal modes (eigenvectors):\n"
            f"{np.array2string(self.modes, precision=6, separator=', ')}\n"
        )


def _build_force_constant_matrix(
    system: System,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    # assert system.n_repeats[1:] == (1, 1), "Only 1D chains are supported."  # noqa: ERA001
    n_x, n_y = system.n_repeats[0], system.n_repeats[1]
    kx, ky = system.spring_constant[0], system.spring_constant[1]
    n = n_x * n_y
    fc = np.zeros((n, n, 3, 3), dtype=float)

    def idx(ix: int, iy: int) -> int:
        return ix * n_y + iy

    for ix in range(n_x):
        for iy in range(n_y):
            i = idx(ix, iy)
            # X direction neighbors
            jx_p = idx((ix + 1) % n_x, iy)
            jx_m = idx((ix - 1) % n_x, iy)
            fc[i, i, 0, 0] += 2 * kx
            fc[i, jx_p, 0, 0] += -kx
            fc[i, jx_m, 0, 0] += -kx
            # Y direction neighbors
            jy_p = idx(ix, (iy + 1) % n_y)
            jy_m = idx(ix, (iy - 1) % n_y)
            fc[i, i, 1, 1] += 2 * ky
            fc[i, jy_p, 1, 1] += -ky
            fc[i, jy_m, 1, 1] += -ky
    return fc


def calculate_normal_modes(system: System) -> NormalModeResult:
    """
    Calculate and plot the normal modes and phonon dispersion relation for a simple 1D chain system.

    Returns a NormalModeResult containing frequencies, eigenvectors, and reduced wave vectors.
    """
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=system.cell,
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = np.diag(system.n_repeats)
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    phonon.force_constants = _build_force_constant_matrix(system)
    phonon.run_mesh(system.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)  # type: ignore[arg-type]
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()  # type: ignore[return-value]

    sorted_indices = np.argsort(mesh_dict["qpoints"][:, 0])  # cspell: disable-line
    return NormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"][sorted_indices] * 1e12 * 2 * np.pi,
        modes=mesh_dict["eigenvectors"][sorted_indices][..., 0],
        q_vals=mesh_dict["qpoints"][sorted_indices, 0],
    )


def plot_dispersion(modes: NormalModeResult) -> tuple[Figure, Axes]:
    """Plot the phonon dispersion relation for a 1D chain on a graph, including analytical curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(modes.q_vals, modes.omega, "o", label="Numerical")

    ax.set_xlim(-0.6, 0.6)
    ax.axvline(0.5, color="gray", linestyle="--", label="First BZ boundary")
    ax.axvline(-0.5, color="gray", linestyle="--")
    ax.axhline(0, color="k", linestyle="-")
    ax.axvline(0, color="k", linestyle="-")

    ax.set_xlabel("Wave vector $q$ (Reduced units)")
    ax.set_ylabel("Frequency $\\omega(q)$")
    ax.set_title("Phonon Dispersion Relation")
    ax.grid(visible=True)
    ax.legend()
    fig.tight_layout()
    return fig, ax
