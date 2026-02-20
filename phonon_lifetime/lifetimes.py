from typing import TYPE_CHECKING

import numpy as np

from phonon_lifetime._util import get_axis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from phonon_lifetime.modes import NormalMode, NormalModes


def get_state_overlap_matrix(
    pristine: NormalModes, defects: NormalModes
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Calculate the overlap matrix S_ki = <d_k | p_i>."""
    states_p = pristine.vectors
    states_d = defects.vectors

    # Assuming states are rows, we take the conjugate of defect states and dot with pristine states
    return np.einsum("kj,ij->ki", states_d.conj(), states_p)


def calculate_survival_probabilities(
    pristine: NormalModes,
    defects: NormalModes,
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
    pristine: NormalMode, defects: NormalModes
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """Calculate the overlap matrix S_ki = <d_k | p_i>."""
    states_d = defects.vectors

    # Assuming states are rows, we take the conjugate of defect states and dot with pristine states
    return np.einsum("kj,j->k", states_d.conj(), pristine.vector.ravel())


def calculate_finite_time_rates(
    pristine: NormalModes, defects: NormalModes, *, t: float
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Calculate the finite-time decay rates of the pristine states after time t."""
    survival_p = calculate_survival_probabilities(
        pristine, defects, times=np.array([t])
    )[:, 0]
    return (1.0 - survival_p) / t


def calculate_survival_probability(
    pristine: NormalMode,
    defects: NormalModes,
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
    pristine: NormalMode,
    defects: NormalModes,
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


def calculate_decay_rates(
    pristine: NormalModes,
    defects: NormalModes,
    *,
    time: float,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Get the decay rate of each pristine state after a time t.

    The transition rate according to first-order perturbation theory is given by Fermi's Golden Rule:
    Gamma_i =  sum_k |<p_k|V|p_i>|^2 / hbar^2 * [ 4 sin^2((omega_k - omega_i) * t / 2) / ((omega_k - omega_i) * t)^2 ]
    where |p_i> is the pristine state, |d_k> are the defect states.
    In the limit of t -> infinity, this reduces to the standard Fermi's Golden Rule with a
    delta function enforcing energy conservation.

    """
    # overlap are W_ki = <\bar{psi}_k| |psi_i>
    # Scatter is the matrix element <\bar{psi}_k|V|p_i> = sum_j W_kj * omega_k (if we ignore k=i)
    overlap = get_state_overlap_matrix(pristine, defects)
    scatter = np.einsum("ki,k->ki", overlap, defects.omega)
    pristine_scatter = np.abs(np.einsum("kf,ki->fi", scatter.conj(), scatter)) ** 2

    delta_omega = pristine.omega[:, np.newaxis] - pristine.omega[np.newaxis, :]
    energy_weight = np.sinc(0.5 * (delta_omega) * time) ** 2

    # |<p_k|V|p_i>|^2 = W_ki * omega_k^2 (if we ignore k=i)
    # We multiply this by an energy_weight for each pristine mode
    rate_per_state = np.einsum("fi,fi->fi", pristine_scatter, energy_weight)
    # Set diagonals to zero to ignore self-scattering, and sum over final states
    np.fill_diagonal(rate_per_state, 0.0)
    return np.sum(rate_per_state, axis=0)
