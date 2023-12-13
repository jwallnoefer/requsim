import numpy as np
from ..noise import NoiseChannel
from ..libs import matrix as mat


def _x_noise_function(rho, epsilon):
    """A single-qubit bit-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.X, rho), mat.H(mat.X))


def _y_noise_function(rho, epsilon):
    """A single-qubit bit-and-phase-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Y, rho), mat.H(mat.Y))


def _z_noise_function(rho, epsilon):
    """A single-qubit phase-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Z, rho), mat.H(mat.Z))


def _w_noise_function(rho, alpha):
    """A single-qubit depolarizing (white) noise channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    alpha : scalar
        Error parameter alpha 0 <= alpha <= 1.
        State is fully depolarized with probability (1-alpha)

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    # trace is necessary if dealing with unnormalized states (e.g. in apply_single_qubit_map)
    return alpha * rho + (1 - alpha) * mat.I(2) / 2 * np.trace(rho)


#: Single-qubit Pauli-X noise channel. Takes error probability `epsilon` as additional argument.
x_noise_channel = NoiseChannel(n_qubits=1, channel_function=_x_noise_function)
#: Single-qubit Pauli-Y noise channel. Takes error probability `epsilon` as additional argument.
y_noise_channel = NoiseChannel(n_qubits=1, channel_function=_y_noise_function)
#: Single-qubit Pauli-Z noise channel. Takes error probability `epsilon` as additional argument.
z_noise_channel = NoiseChannel(n_qubits=1, channel_function=_z_noise_function)
#: Single-qubit white noise (=fully depolarizing) channel. Takes error parameter `alpha` as additional argument.
w_noise_channel = NoiseChannel(n_qubits=1, channel_function=_w_noise_function)
