"""A collection of code for manipulating normal modes."""

from ._calculate import calculate_normal_modes
from ._modes import (
    NormalMode,
    NormalModeResult,
    PristineMode,
    PristineNormalModeResult,
    VacancyMode,
    VacancyNormalModeResult,
)
from ._plot import (
    animate_mode_1d_x,
    animate_mode_2d_xy,
    plot_1d_dispersion,
    plot_dispersion_2d_xy,
    plot_mode_1d_x,
    plot_mode_2d_xy,
)

__all__ = [
    "NormalMode",
    "NormalModeResult",
    "PristineMode",
    "PristineNormalModeResult",
    "VacancyMode",
    "VacancyNormalModeResult",
    "animate_mode_1d_x",
    "animate_mode_2d_xy",
    "calculate_normal_modes",
    "plot_1d_dispersion",
    "plot_dispersion_2d_xy",
    "plot_mode_1d_x",
    "plot_mode_2d_xy",
]
