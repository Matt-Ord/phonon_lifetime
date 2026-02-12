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
class NormalMode:
    omega: float
    """The normal mode frequencies in angular frequency units."""
    modes: np.array
    """The eigenvectors (normal modes) of the system."""
    q_val: np.ndarray
    """The reduced wave vectors for the normal modes."""


@dataclass(kw_only=True, frozen=True)
class ModesAtBranch:
    omega: np.ndarray[Any, np.dtype[np.floating]]
    """The normal mode frequencies in angular frequency units."""
    modes: np.ndarray[Any, np.dtype[np.floating]]
    """The eigenvectors (normal modes) of the system."""
    q_vals: np.ndarray[Any, np.dtype[np.floating]]
    """The reduced wave vectors for the normal modes."""


@dataclass(kw_only=True, frozen=True)
class ModesAtQ:
    omega: np.ndarray[Any, np.dtype[np.floating]]
    """The normal mode frequencies in angular frequency units."""
    modes: np.ndarray[Any, np.dtype[np.floating]]
    """The eigenvectors (normal modes) of the system."""
    q_val: np.ndarray[Any, np.dtype[np.floating]]
    """The reduced wave vectors for the normal modes."""
    getmodesatbranch


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

    def get_ModesAtBranch(self, branch):
        return self.modes[..., branch]

    def get_ModesAtQPoint(self, q_index):
        return self.modes[q_index, ...]

    @property
    def q_x(self) -> np.ndarray[Any]:
        return self.q_vals[..., 0]

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
        modes=mesh_dict["eigenvectors"][sorted_indices],
        q_vals=mesh_dict["qpoints"][sorted_indices, 0],
    )


def plot_dispersion(modes: NormalModeResult) -> tuple[Figure, Axes]:
    """Plot the phonon dispersion relation for a 1D chain on a graph, including analytical
    curve.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        np.fft.ifftshift(modes.q_vals),
        np.fft.ifftshift(modes.dispersion),
        "o-",
        label="Dispersion relation",
    )
    ax.set_xlim(
        -np.pi / modes.system.lattice_constant[0],
        np.pi / modes.system.lattice_constant[0],
    )
    ax.set_xlabel("Wave vector $q$")
    ax.set_ylabel("Frequency $\\omega(q)$")
    ax.set_title("Phonon Dispersion Relation")
    ax.grid(visible=True)
    ax.legend()
    fig.tight_layout()
    return fig, ax

    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(modes.q_x, modes.omega, "o-", label="Numerical")

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
"""
    return None


"""
Below are 2-D code
"""


@dataclass(kw_only=True, frozen=True)
class SquareLattice2DSystem:
    """Represents a 2D square lattice system for phonon calculations.

    Attributes
    ----------
    element : str
        The chemical symbol of the element.
    lattice_constantx : float
        Lattice constant along the x direction.
    lattice_constanty : float
        Lattice constant along the y direction.
    n_repeatsx : int
        Number of repeats along the x direction.
    n_repeatsy : int
        Number of repeats along the y direction.
    k_nn : float
        Nearest neighbor spring constant.
    k_nnn : float
        Next nearest neighbor spring constant.
    """

    element: str
    lattice_constantx: float
    lattice_constanty: float
    n_repeatsx: int
    n_repeatsy: int
    k_nn: float
    k_nnn: float

    @property
    def mass(self) -> float:
        """Return the mass of the element in atomic mass units."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=[
                [self.lattice_constantx, 0, 0],
                [0, self.lattice_constanty, 0],
                [0, 0, 1],
            ],
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


def build_force_constants_2d(system: SquareLattice2DSystem) -> np.ndarray:
    """
    Build the force constant matrix for a 2D square lattice system including nearest and next nearest neighbor interactions.

    Parameters
    ----------
    system : SquareLattice2DSystem
        The 2D square lattice system for which to build the force constant matrix.

    Returns
    -------
    np.ndarray
        The force constant matrix of shape (num_atoms, num_atoms, 3, 3).
    """
    nx = system.n_repeatsx
    ny = system.n_repeatsy
    num_atoms = nx * ny
    fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=[
            [system.lattice_constantx, 0, 0],
            [0, system.lattice_constanty, 0],
            [0, 0, 1],
        ],
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, 1]]
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
    position = phonon.supercell.positions

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            vec: np.ndarray = position[j] - position[i]
            vec -= np.round(
                vec
                / np.array(
                    [system.lattice_constantx * nx, system.lattice_constanty * ny, 1]
                )  # Periodic boundary conditions
            ) * np.array(
                [system.lattice_constantx * nx, system.lattice_constanty * ny, 1]
            )
            dist = np.linalg.norm(vec[:2])
            a = max(system.lattice_constantx, system.lattice_constanty)
            if np.isclose(dist, a, atol=0.05):  # Neartest-neighbor
                direction = vec / np.linalg.norm(vec)
                for d1 in range(3):
                    for d2 in range(3):
                        fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
                        fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]
            elif np.isclose(dist, np.sqrt(2) * a, atol=0.05):  # Next-nearest-neighbor
                direction = vec / np.linalg.norm(vec)
                for d1 in range(3):
                    for d2 in range(3):
                        fc[i, j, d1, d2] += (
                            -system.k_nnn * direction[d1] * direction[d2]
                        )
                        fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
    return fc
