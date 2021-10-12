"""Functions for pre-defined entanglement purification protocols."""
import numpy as np
from . import matrix as mat

sqrt_plus_ix = 1 / np.sqrt(2) * (mat.I(2) + 1j * mat.X)
sqrt_minus_ix = mat.H(sqrt_plus_ix)
bilateral_cnot = np.dot(mat.CNOT(0, 2, N=4), mat.CNOT(1, 3, N=4))
dejmps_operator = np.dot(
    bilateral_cnot, mat.tensor(sqrt_minus_ix, sqrt_plus_ix, sqrt_minus_ix, sqrt_plus_ix)
)
dejmps_operator_dagger = mat.H(dejmps_operator)
dejmps_proj_ket_z0z0 = mat.tensor(mat.I(4), mat.z0, mat.z0)
dejmps_proj_ket_z1z1 = mat.tensor(mat.I(4), mat.z1, mat.z1)
dejmps_proj_bra_z0z0 = mat.H(dejmps_proj_ket_z0z0)
dejmps_proj_bra_z1z1 = mat.H(dejmps_proj_ket_z1z1)


def dejmps_protocol(rho):
    """Applies the DEJMPS entanglement purification protocol.

    Input is usually two entangled pairs and output is one entangled pair if
    successful.
    This protocol was introduced in:
    D. Deutsch, et. al., Phys. Rev. Lett., vol. 77, pp. 2818–2821 (1996)
    arXiv:quant-ph/9604039


    Parameters
    ----------
    rho : np.ndarray
        Four-qubit density matrix (16x16).

    Returns
    -------
    p_suc : scalar
        probability of success for the protocol
    state : np.ndarray
        Two-qubit density matrix (4x4). The state of the remaining pair IF the
        protocol was successful.
    """
    rho = np.dot(np.dot(dejmps_operator, rho), dejmps_operator_dagger)
    rho = np.dot(np.dot(dejmps_proj_bra_z0z0, rho), dejmps_proj_ket_z0z0) + np.dot(
        np.dot(dejmps_proj_bra_z1z1, rho), dejmps_proj_ket_z1z1
    )
    p_suc = np.trace(rho)
    state = rho / p_suc  # renormalize
    return p_suc, state
