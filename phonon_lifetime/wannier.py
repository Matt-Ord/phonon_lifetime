from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from phonon_lifetime._util import get_axis
from phonon_lifetime.system import get_atom_centres

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from phonon_lifetime.modes import NormalModes


def _get_wannier_vectors(
    modes: NormalModes,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
]:
    """Get the modes of the system using Wannier interpolation."""
    phonon_vectors = modes.vectors
    omega = modes.omega

    q, _r, _p = scipy.linalg.qr(phonon_vectors, pivoting=True)

    wannier_vectors = q.conj().T @ phonon_vectors
    h_wannier = q.conj().T @ (omega[:, np.newaxis] * q)

    return wannier_vectors, h_wannier


def plot_wannier_vector(
    modes: NormalModes,
    idx: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Wannier vectors of the system."""
    fig, ax = get_axis(ax)

    wannier_vectors, _h_wannier = _get_wannier_vectors(modes)
    wannier_vectors = wannier_vectors.reshape(modes.n_modes, -1, 3)

    for i in range(3):
        state = wannier_vectors[idx, :, i]

        ax.plot(
            get_atom_centres(modes.system)[:, 0],
            np.real(state),
            label=f"Wannier {idx} in {'xyz'[i]}",
        )
    ax.set_title("Wannier Vectors")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    return fig, ax
