from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, override

import numpy as np

from phonon_lifetime.phonon._phonon import Phonons, get_phonons
from phonon_lifetime.system import StrainSystem

if TYPE_CHECKING:
    from collections.abc import Iterator


type DispersionPoint = tuple[str, tuple[float, float, float]]

type DispersionPathPoint = tuple[int, DispersionPoint]


@dataclass(frozen=True, kw_only=True)
class DispersionPath:
    """A path through the Brillouin zone for plotting the dispersion relation."""

    points: tuple[DispersionPoint, *tuple[DispersionPathPoint, ...]]

    @property
    def _remaining_points(self) -> tuple[DispersionPathPoint, ...]:
        """Get the points in the path, excluding the first one."""
        return self.points[1:]

    def __iter__(self) -> Iterator[DispersionSegment]:
        start, *_rest = self.points
        for point in self._remaining_points:
            yield DispersionSegment(
                start=start,
                end=point[1],
                n_points=point[0],
            )
            start = point[1]

    @property
    def q_values(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        """The q values for the dispersion path."""
        out = []
        for segment in self:
            q_points = segment.q_values
            out.append(q_points)
        return np.concatenate(out, axis=0)

    @property
    def labels(self) -> tuple[str, ...]:
        """The labels for the high-symmetry points in the path."""
        return (self.points[0][0], *(point[1][0] for point in self._remaining_points))


@dataclass(frozen=True, kw_only=True)
class DispersionSegment:
    """A segment of the dispersion path."""

    start: DispersionPoint
    end: DispersionPoint
    n_points: int

    @property
    def q_values(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        """The n q values for this segment."""
        start_q = np.array(self.start[1])
        end_q = np.array(self.end[1])
        return np.linspace(start_q, end_q, self.n_points)


class HexagonalPoint(Enum):
    """High-symmetry points for 2D and 3D hexagonal lattices (e.g., Graphene, HCP)."""

    GAMMA = ("Γ", (0.0, 0.0, 0.0))
    M = ("M", (0.5, 0.0, 0.0))
    MINUS_M = ("-M", (-0.5, 0.0, 0.0))
    K = ("K", (1 / 3, 1 / 3, 0.0))
    MINUS_K = ("-K", (-1 / 3, -1 / 3, 0.0))
    A = ("A", (0.0, 0.0, 0.5))  # 3D specific
    MINUS_A = ("-A", (0.0, 0.0, -0.5))  # 3D specific
    L = ("L", (0.5, 0.0, 0.5))  # 3D specific
    MINUS_L = ("-L", (-0.5, 0.0, -0.5))  # 3D specific
    H = ("H", (1 / 3, 1 / 3, 0.5))  # 3D specific
    MINUS_H = ("-H", (-1 / 3, -1 / 3, -0.5))  # 3D specific


class CubicPoint(Enum):
    """High-symmetry points for a Cubic lattice."""

    GAMMA = ("Γ", (0.0, 0.0, 0.0))
    X = ("X", (0.5, 0.0, 0.0))
    MINUS_X = ("-X", (-0.5, 0.0, 0.0))
    M = ("M", (0.5, 0.5, 0.0))
    MINUS_M = ("-M", (-0.5, -0.5, 0.0))
    R = ("R", (0.5, 0.5, 0.5))
    MINUS_R = ("-R", (-0.5, -0.5, -0.5))
    Y = ("Y", (0.0, 0.5, 0.0))
    MINUS_Y = ("-Y", (0.0, -0.5, 0.0))
    Z = ("Z", (0.0, 0.0, 0.5))
    MINUS_Z = ("-Z", (0.0, 0.0, -0.5))


class FaceCenteredCubicPoint(Enum):
    """High-symmetry points for an FCC lattice (Standard Setyawan-Curtarolo convention)."""

    GAMMA = ("Γ", (0.0, 0.0, 0.0))
    X = ("X", (0.5, 0.0, 0.5))
    L = ("L", (0.5, 0.5, 0.5))
    W = ("W", (0.5, 0.25, 0.75))
    K = ("K", (0.375, 0.375, 0.75))
    U = ("U", (0.625, 0.25, 0.625))


class BodyCenteredCubicPoint(Enum):
    """High-symmetry points for a BCC lattice."""

    GAMMA = ("Γ", (0.0, 0.0, 0.0))
    H = ("H", (0.5, -0.5, 0.5))
    P = ("P", (0.25, 0.25, 0.25))
    N = ("N", (0.0, 0.0, 0.5))


class DispersionPathPhonons[S: StrainSystem = StrainSystem](Phonons[S]):
    """A collection of phonon modes that form a dispersion relation."""

    def __init__(
        self,
        system: S,
        omega: np.ndarray[tuple[int], np.dtype[np.floating]],
        vectors: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]],
        path: DispersionPath,
    ) -> None:
        self._system = system
        self._omega = omega
        self._vectors = vectors
        self._path = path

    @property
    @override
    def q_values(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        """The q values for each phonon."""
        return self._path.q_values

    @property
    def path(self) -> DispersionPath:
        """The path through the Brillouin zone that these phonons correspond to."""
        return self._path


def get_dispersion_path[S: StrainSystem = StrainSystem](
    system: S, path: DispersionPath
) -> DispersionPathPhonons[S]:
    """Get the normal modes of the system."""
    phonons = get_phonons(system, q_values=path.q_values)

    return DispersionPathPhonons(
        system=system,
        omega=phonons.omega,
        vectors=phonons.vectors,
        path=path,
    )


class DispersionSegmentPhonons[S: StrainSystem = StrainSystem](Phonons[S]):
    """A collection of phonon modes that form a dispersion relation."""

    def __init__(
        self,
        system: S,
        omega: np.ndarray[tuple[int], np.dtype[np.floating]],
        vectors: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.complex128]],
        path: DispersionSegment,
    ) -> None:
        self._system = system
        self._omega = omega
        self._vectors = vectors
        self._segment = path

    @property
    @override
    def q_values(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        """The q values for each phonon."""
        return self._segment.q_values

    @property
    def segment(self) -> DispersionSegment:
        """The segment of the dispersion path that these phonons correspond to."""
        return self._segment


def get_dispersion_segment[S: StrainSystem = StrainSystem](
    system: S, segment: DispersionSegment
) -> DispersionSegmentPhonons[S]:
    """Get the normal modes of the system."""
    phonons = get_phonons(system, q_values=segment.q_values)

    return DispersionSegmentPhonons(
        system=system,
        omega=phonons.omega,
        vectors=phonons.vectors,
        path=segment,
    )
