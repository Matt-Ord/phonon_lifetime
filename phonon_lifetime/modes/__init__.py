"""A collection of code for manipulating normal modes."""

from ._mode import CanonicalMode, NormalMode, NormalModes
from ._plot import (
    animate_mode_1d_x,
    animate_mode_2d_xy,
    animate_mode_xyz,
    plot_mode_1d_x,
    plot_mode_2d_xy,
    plot_mode_xyz,
)
from ._util import get_mode_displacement, repeat_mode, repeat_modes

__all__ = [
    "CanonicalMode",
    "NormalMode",
    "NormalModes",
    "animate_mode_1d_x",
    "animate_mode_2d_xy",
    "animate_mode_xyz",
    "get_mode_displacement",
    "plot_mode_1d_x",
    "plot_mode_2d_xy",
    "plot_mode_xyz",
    "repeat_mode",
    "repeat_modes",
]
