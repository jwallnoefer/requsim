import os
import numpy as np
from ..libs import matrix as mat
import pandas as pd
from warnings import warn


def binary_entropy(p):
    if p == 1 or p == 0:
        return 0
    else:
        res = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        if np.isnan(res):
            warn(f"binary_entropy was called with p={p} and returned nan")
        return res


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
