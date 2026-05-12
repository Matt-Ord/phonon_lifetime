from typing import TYPE_CHECKING

import numpy as np

from phonon_lifetime._util import get_axis
from phonon_lifetime.phonon import MeshPhonons, as_gamma_phonons

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from phonon_lifetime.phonon import GammaPhonon, GammaPhonons


def _assert_same_cell(
    pristine: GammaPhonons | GammaPhonon, defects: GammaPhonons
) -> None:
    assert pristine.system.cell == defects.system.cell, (
        "Pristine and defect systems must have the same cell"
    )


def get_state_overlap_matrix(
    pristine: GammaPhonons, defects: GammaPhonons
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Calculate the overlap matrix S_ki = <d_k | p_i>."""
    _assert_same_cell(pristine, defects)
    states_p = pristine.vectors
    states_d = defects.vectors

    # Assuming states are rows, we take the conjugate of defect states and dot with pristine states
    return np.einsum("iab,jab->ij", states_d.conj(), states_p)


def calculate_survival_probabilities(
    pristine: GammaPhonons,
    defects: GammaPhonons,
    *,
    times: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Get the survival probability of each pristine state after a time t.

    returns an array of shape (n_pristine, n_times) where each element is the probability that the pristine state has not decayed after time t.
    """
    # weights are W_ki = <psi_i| |\bar{psi}_k><\bar{psi}_k| |psi_i> = sum_k |<\bar{psi}_k|psi_i>|^2
    overlap = get_state_overlap_matrix(pristine, defects)
    weights = np.abs(overlap) ** 2

    # The total overlap <psi_i| e^{-iHt} |psi_i>
    # is the same as sum_k  <psi_i| |\bar{psi}_k><\bar{psi}_k| e^{-iHt} |psi_i>
    survival_amplitude = np.einsum(
        "ki,kj->ij",
        weights,
        np.exp(-1j * defects.omega[:, np.newaxis] * times[np.newaxis, :]),
    )

    # Probability is the square of the amplitude
    return np.abs(survival_amplitude) ** 2


def get_state_overlap(
    pristine: GammaPhonon, defects: GammaPhonons
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """Calculate the overlap matrix S_ki = <d_k | p_i>."""
    _assert_same_cell(pristine, defects)
    states_d = defects.vectors

    # Assuming states are rows, we take the conjugate of defect states and dot with pristine states
    return np.einsum("ijk,jk->i", states_d.conj(), pristine.vector)


def calculate_finite_time_rates(
    pristine: GammaPhonons, defects: GammaPhonons, *, t: float
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Calculate the finite-time decay rates of the pristine states after time t."""
    survival_p = calculate_survival_probabilities(
        pristine, defects, times=np.array([t])
    )[:, 0]
    return (1.0 - survival_p) / t


def calculate_survival_probability(
    pristine: GammaPhonon,
    defects: GammaPhonons,
    *,
    times: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Get the survival probability of a pristine state after a time t.

    returns an array of shape (n_times) where each element is the probability that the pristine state has not decayed after time t.
    """
    # weights are W_k = <psi_i| |\bar{psi}_k><\bar{psi}_k| |psi_i> = sum_k |<\bar{psi}_k|psi_i>|^2
    # The probability of existing in a particular defect mode
    overlap = get_state_overlap(pristine, defects)
    weights = np.abs(overlap) ** 2

    # The total overlap <psi_i| e^{-iHt} |psi_i>
    # is the same as sum_k  <psi_i| |\bar{psi}_k><\bar{psi}_k| e^{-iHt} |psi_i>
    survival_amplitude = np.einsum(
        "k,kj->j",
        weights,
        np.exp(-1j * defects.omega[:, np.newaxis] * times[np.newaxis, :]),
    )

    # Probability is the square of the amplitude
    return np.abs(survival_amplitude) ** 2


def plot_survival_probability(
    pristine: GammaPhonon,
    defects: GammaPhonons,
    times: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the survival probabilities of the pristine states after time t."""
    fig, ax = get_axis(ax)
    survival_p = calculate_survival_probability(pristine, defects, times=times)

    (line,) = ax.plot(times, survival_p)
    ax.set_title("Survival Probabilities against time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    return fig, ax, line


def plot_overlap_weights(
    pristine: GammaPhonon,
    defects: GammaPhonons,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    r"""Plot the overlap weights of a pristine state with the defect states against the defect frequencies.

    for a pristine state |psi_i>, the overlap weight with a defect mode k is given by
    $W_(omega_k)^i = |<\bar{psi}_k|psi_i>|^2$ where omega_k is the frequency of the defect mode k.

    """
    fig, ax = get_axis(ax)
    overlap = get_state_overlap(pristine, defects)
    weights = np.abs(overlap) ** 2

    (line,) = ax.plot(defects.omega, weights)
    line.set_marker("x")
    ax.axvline(pristine.omega, linestyle="--", label="Pristine Frequency")

    ax.set_title("Overlap Weights against defect frequency")
    ax.set_xlabel("Defect Frequency")
    ax.set_ylabel("Overlap Weight")
    ax.set_ylim(0, None)
    ax.legend()
    return fig, ax, line


def get_first_order_scatter(
    pristine: GammaPhonons,
    defects: GammaPhonons,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Calculate the first-order scattering matrix element <p_k|V|p_i>."""
    # Scatter is
    # <p_k|V|p_i> = sum_j <p_k|d_j><d_j| H_def - H_pristine |p_i>
    #             = sum_j <p_k|d_j><d_j|p_i> (omega^d_j - omega_i)
    #             = sum_j W_jk^*  W_ji (omega^d_j - omega_i)
    # overlap are W_ki = <d_k| |p_i>
    overlap = get_state_overlap_matrix(pristine, defects)
    d_omega = defects.omega[:, np.newaxis] - pristine.omega[np.newaxis, :]
    return np.einsum("jk,ji,ji->ki", np.conj(overlap), overlap, d_omega)


def plot_first_order_scatter(
    pristine: GammaPhonons,
    defects: GammaPhonons,
    pristine_idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    r"""Plot the first-order scattering matrix element <p_k|V|p_i> against the defect frequencies.

    for a pristine state |psi_i>, the first-order scattering with a defect mode k is given by
    $S_(omega_k)^i = |<\bar{psi}_k|V|psi_i>|^2$ where omega_k is the frequency of the defect mode k.

    """
    fig, ax = get_axis(ax)

    scatter = get_first_order_scatter(pristine, defects)
    weights = np.abs(scatter[:, pristine_idx]) ** 2

    (line,) = ax.plot(pristine.omega, weights)
    line.set_marker("x")
    line.set_linestyle("")
    ax.axvline(pristine[pristine_idx].omega, linestyle="--", label="Pristine Frequency")

    ax.set_title("First-order Scattering against defect frequency")
    ax.set_xlabel("State Frequency")
    ax.set_ylabel("Scattering Strength")
    ax.set_ylim(0, None)
    return fig, ax, line


def plot_first_order_scatter_against_qx(
    pristine: MeshPhonons,
    defects: GammaPhonons,
    pristine_idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    r"""Plot the first-order scattering matrix element <p_k|V|p_i> against the pristine q.

    for a pristine state |psi_i>, the first-order scattering with a defect mode k is given by
    $S_(omega_k)^i = |<\bar{psi}_k|V|psi_i>|^2$ where omega_k is the frequency of the defect mode k.

    """
    fig, ax = get_axis(ax)

    scatter = get_first_order_scatter(as_gamma_phonons(pristine), defects)
    weights = np.abs(scatter[:, pristine_idx]) ** 2

    for band in range(pristine.n_branch):
        band_indices = pristine.get_mode_idx(branch=band)
        (line,) = ax.plot(
            np.fft.fftshift(pristine.q_values[:, 0]),
            np.fft.fftshift(weights[band_indices]),
        )
        line.set_marker("x")

    line = ax.axvline(pristine[pristine_idx].q[0], linestyle="--")
    line.set_label("Pristine qx")

    ax.legend()
    ax.set_title("First-order Scattering against qx")
    ax.set_xlabel("State Frequency")
    ax.set_ylabel("Scattering Strength")
    ax.set_ylim(0, None)
    return fig, ax, line
