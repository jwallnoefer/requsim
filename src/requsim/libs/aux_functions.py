import os
import numpy as np
from . import matrix as mat
import pandas as pd
from warnings import warn


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


def binary_entropy(p):
    if p == 1 or p == 0:
        return 0
    else:
        res = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        if np.isnan(res):
            warn(f"binary_entropy was called with p={p} and returned nan")
        return res


def distance(pos1, pos2):
    """Return the euclidean distance between two positions or world objects.

    Parameters
    ----------
    pos1 : scalar, np.ndarray or WorldObject
        The first position. If it is a WorldObject must have a position attribute.
    pos2 : scalar, np.ndarray or WorldObject
        The second position. If it is a WorldObject must have a position attribute.

    Returns
    -------
    scalar
        Distance between the world objects.

    """
    try:
        pos1 = pos1.position
    except AttributeError:
        pass
    try:
        pos2 = pos2.position
    except AttributeError:
        pass
    if np.isscalar(pos1) and np.isscalar(pos2):
        return np.abs(pos1 - pos2)
    elif isinstance(pos1, np.ndarray) and isinstance(pos2, np.ndarray):
        if pos1.shape == pos2.shape:
            return np.sqrt(np.sum((pos1 - pos2) ** 2))
        else:
            ValueError(
                f"Can't calculate distance between positions with shape {pos1.shape} and {pos2.shape}"
            )
    else:
        raise TypeError(
            f"Can't calculate distance between positions of type {type(pos1)} and type {type(pos2)}"
        )


def calculate_keyrate_time(
    correlations_z, correlations_x, err_corr_ineff, time_interval, return_std=False
):
    e_z = 1 - np.mean(correlations_z)
    e_x = 1 - np.mean(correlations_x)
    pair_per_time = len(correlations_z) / time_interval
    keyrate = pair_per_time * (
        1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z)
    )
    if not return_std:
        return keyrate
    # use error propagation formula
    if e_z == 0:
        keyrate_std = pair_per_time * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
        )
    else:
        keyrate_std = pair_per_time * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
            + err_corr_ineff ** 2
            * (-np.log2(e_z) + np.log2(1 - e_z)) ** 2
            * np.std(correlations_z) ** 2
        )
    return keyrate, keyrate_std


def calculate_keyrate_channel_use(
    correlations_z, correlations_x, err_corr_ineff, resource_list, return_std=False
):
    e_z = 1 - np.mean(correlations_z)
    e_x = 1 - np.mean(correlations_x)
    pair_per_resource = len(correlations_z) / np.sum(resource_list)
    keyrate = pair_per_resource * (
        1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z)
    )
    if not return_std:
        return keyrate
    # use error propagation formula
    if e_z == 0:
        keyrate_std = pair_per_resource * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
        )
    else:
        keyrate_std = pair_per_resource * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
            + err_corr_ineff ** 2
            * (-np.log2(e_z) + np.log2(1 - e_z)) ** 2
            * np.std(correlations_z) ** 2
        )
    return keyrate, keyrate_std


def standard_bipartite_evaluation(data_frame, err_corr_ineff=1):
    states = data_frame["state"]

    fidelity_list = np.real_if_close(
        [
            np.dot(np.dot(mat.H(mat.phiplus), state), mat.phiplus)[0, 0]
            for state in states
        ]
    )
    fidelity = np.mean(fidelity_list)
    fidelity_std = np.std(fidelity_list)

    z0z0 = mat.tensor(mat.z0, mat.z0)
    z1z1 = mat.tensor(mat.z1, mat.z1)
    correlations_z = np.real_if_close(
        [
            np.dot(np.dot(mat.H(z0z0), state), z0z0)[0, 0]
            + np.dot(np.dot(mat.H(z1z1), state), z1z1)[0, 0]
            for state in states
        ]
    )
    correlations_z[correlations_z > 1] = 1

    x0x0 = mat.tensor(mat.x0, mat.x0)
    x1x1 = mat.tensor(mat.x1, mat.x1)
    correlations_x = np.real_if_close(
        [
            np.dot(np.dot(mat.H(x0x0), state), x0x0)[0, 0]
            + np.dot(np.dot(mat.H(x1x1), state), x1x1)[0, 0]
            for state in states
        ]
    )
    correlations_x[correlations_x > 1] = 1

    key_per_time, key_per_time_std = calculate_keyrate_time(
        correlations_z=correlations_z,
        correlations_x=correlations_x,
        err_corr_ineff=err_corr_ineff,
        time_interval=data_frame["time"].iloc[-1],
        return_std=True,
    )
    key_per_resource, key_per_resource_std = calculate_keyrate_channel_use(
        correlations_z=correlations_z,
        correlations_x=correlations_x,
        err_corr_ineff=err_corr_ineff,
        resource_list=data_frame["resource_cost_max"],
        return_std=True,
    )
    return [
        fidelity,
        fidelity_std,
        key_per_time,
        key_per_time_std,
        key_per_resource,
        key_per_resource_std,
    ]


def save_result(data_series, output_path, mode="write"):
    """Evaluate and save data in a standardized way.

    Parameters
    ----------
    data_series : pandas.Series
        A Series of pandas.DataFrame retrieved via protocol.data,
        index should be the x-axis of the corresponding plot.
    output_path : str
        Results are written to this path..
    mode : {"write", "w", "append", "a"}
        If "write" or "w" overwrites existing results.
        If "append" or "a" will look for existing results and append the results
        if there are any.

    Returns
    -------
    None

    """
    if mode in ["write", "w"]:
        append_mode = False
    elif mode in ["append", "a"]:
        append_mode = True
    else:
        raise ValueError(
            f"save_result does not support mode {mode}, please choose either 'write' or 'append'"
        )
    assert_dir(output_path)
    result_list = [standard_bipartite_evaluation(data_frame=df) for df in data_series]
    output_data = pd.DataFrame(
        data=result_list,
        index=data_series.index,
        columns=[
            "fidelity",
            "fidelity_std",
            "key_per_time",
            "key_per_time_std",
            "key_per_resource",
            "key_per_resource_std",
        ],
    )
    if append_mode:
        try:
            existing_series = pd.read_pickle(os.path.join(output_path, "raw_data.bz2"))
            data_series = existing_series.append(data_series)
        except FileNotFoundError:
            pass
        try:
            existing_data = pd.read_csv(
                os.path.join(output_path, "result.csv"), index_col=0
            )
            output_data = pd.concat([existing_data, output_data])
        except FileNotFoundError:
            pass
    data_series.to_pickle(os.path.join(output_path, "raw_data.bz2"))
    output_data.to_csv(os.path.join(output_path, "result.csv"))


def assert_dir(path):
    """Check if `path` exists, and create it if it doesn't.

    Parameters
    ----------
    path : str
        The path to be checked/created.

    Returns
    -------
    None

    """
    if not os.path.exists(path):
        os.makedirs(path)


def apply_single_qubit_map(map_func, qubit_index, rho, *args, **kwargs):
    """Applies a single-qubit map to a density matrix of n qubits.

    Parameters
    ----------
    map_func : callable
        The map to apply. Should be a function that takes a single-qubit density
        matrix as input and applies the map to it.
    qubit_index : int
        Index of qubit to which the map is applied. 0...n-1
    rho : np.ndarray
        Density matrix of n qubits. Shape (2**n, 2**n)
    *args, **kwargs: any, optional
        additional args and kwargs passed to map_func

    Returns
    -------
    np.ndarray
        The density matrix with the map applied. Shape (2**n, 2**n)

    """
    n = int(np.log2(rho.shape[0]))
    rho = rho.reshape((2, 2) * n)
    # there must be a nicer way to do the iteration here:
    out = np.zeros(rho.shape, dtype=complex)
    for idx in np.ndindex(*(2, 2) * (n - 1)):
        my_slice = (
            idx[:qubit_index]
            + (slice(None),)
            + idx[qubit_index : n - 1 + qubit_index]
            + (slice(None),)
            + idx[n - 1 + qubit_index :]
        )
        out[my_slice] = map_func(rho[my_slice], *args, **kwargs)
    return np.real_if_close(out.reshape((2 ** n, 2 ** n)))


def apply_m_qubit_map(map_func, qubit_indices, rho, *args, **kwargs):
    """Applies an m-qubit map to a density matrix of n qubits.

    Parameters
    ----------
    map_func : callable
        The map to apply. Should be a function that takes a single-qubit density
        matrix as input and applies the map to it.
    qubit_indices : list of ints
        Indices of qubit to which the map is applied. Indices from 0...n-1
    rho : np.ndarray
        Density matrix of n qubits. Shape (2**n, 2**n)
    *args, **kwargs: any, optional
        additional args and kwargs passed to map_func

    Returns
    -------
    np.ndarray
        The density matrix with the map applied. Shape (2**n, 2**n)

    """
    m = len(qubit_indices)
    # if m == 1:
    #     return apply_single_qubit_map(map_func=map_func, qubit_index=qubit_indices[0], rho=rho, *args, **kwargs)
    n = int(np.log2(rho.shape[0]))
    rho = rho.reshape((2, 2) * n)
    assert m <= n
    qubit_indices = sorted(qubit_indices)
    index_list = qubit_indices + [n + qubit_index for qubit_index in qubit_indices]
    # still not found a nicer way for the iteration here
    out = np.zeros_like(rho)
    for idx in np.ndindex(*(2, 2) * (n - m)):
        my_slice = list(idx)
        for current_idx in index_list:
            my_slice.insert(current_idx, slice(None))
        my_slice = tuple(my_slice)
        # print(idx, n, m, qubit_indices, index_list)
        # print(my_slice)
        out[my_slice] = map_func(
            rho[my_slice].reshape(2 ** m, 2 ** m), *args, **kwargs
        ).reshape((2, 2) * m)
    return out.reshape((2 ** n, 2 ** n))


# def apply_m_qubit_map_alternate(map_func, qubit_indices, rho, *args, **kwargs):
#     m = len(qubit_indices)
#     n = int(np.log2(rho.shape[0]))
#     rho = rho.reshape((2, 2) * n)
#     assert m <= n
#     qubit_indices = sorted(qubit_indices)
#     index_list = qubit_indices + [n + qubit_index for qubit_index in qubit_indices]
#     perm_list = [i for i in range(2 * n)]
#     unperm_list = [i for i in range(2 * (n - m))]
#     for j, current_idx in enumerate(index_list):
#         perm_list.remove(current_idx)
#         perm_list += [current_idx]
#         unperm_list.insert(current_idx, 2 * (n - m) + j)
#     rho = rho.transpose(perm_list).reshape((2, 2) * (n - m) + (2**m, 2**m))
#     map_func = np.vectorize(map_func, signature="(i,j)->(i,j)")
#     out = map_func(rho).reshape((2, 2) * n)
#     # print(n, m, qubit_indices, index_list)
#     # print(perm_list, unperm_list)
#     return out.transpose(unperm_list).reshape((2**n, 2**n))


def x_noise_channel(rho, epsilon):
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


def y_noise_channel(rho, epsilon):
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


def z_noise_channel(rho, epsilon):
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


def w_noise_channel(rho, alpha):
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
    return alpha * rho + (1 - alpha) * mat.I(2) / 2 * np.trace(
        rho
    )  # trace is necessary if dealing with unnormalized states (e.g. in apply_single_qubit_map)


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
