import os
import numpy as np
from ..libs import matrix as mat
import pandas as pd
from warnings import warn


def binary_entropy(p):
    """Calculate the binary entropy.

    Parameters
    ----------
    p : scalar
        Must be in interval [0, 1]. Usually an error rate.

    Returns
    -------
    scalar
        The binary entropy of `p`.

    """
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
    """Calculate the asymptotic key rate per time from a list of correlations.

    This uses the sample mean of error rates to estimate what the asymptotic
    key rate would be. It uses the formula for a bound on the asymptotic key
    rate. See for this particular formulation:

    D. Luong, L. Jiang, J. Kim, N. Lütkenhaus; Appl. Phys. B 122, 96 (2016)
    arXiv:1508.02811 [quant-ph]

    Optionally also returns the standard deviation of the key rate
    calculated by using propagation of error.

    Parameters
    ----------
    correlations_z : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in z-direction coincide for each raw bit.
    correlations_x : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in x-direction coincide for each raw bit.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies
    time_interval : scalar
        Time interval in which the raw bits were collected.
    return_std : bool
        Whether to also compute and return the standard deviation of the
        key rate. Default: False

    Returns
    -------
    scalar or tuple of scalars
        If return_std is False, returns the key rate.
        If return_std is True, returns at tuple of key rate and
        standard deviation of the key rate.

    """
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
            + err_corr_ineff**2
            * (-np.log2(e_z) + np.log2(1 - e_z)) ** 2
            * np.std(correlations_z) ** 2
        )
    return keyrate, keyrate_std


def calculate_keyrate_channel_use(
    correlations_z, correlations_x, err_corr_ineff, resource_list, return_std=False
):
    """Calculate the asymptotic key rate per resource from a list of correlations.

    CAREFUL: This formulation only makes sense when the amount of resources
    (usually number of channel uses or similar) is directly assignable to one
    particular pair or set of raw bits. This is often not the case e.g. if
    connections are not established sequentially for each bit, as would be the
    case for multi-mode memories.
    This function uses the sample mean of error rates to estimate what the
    asymptotic key rate would be.
    It uses the formula for a bound on the asymptotic key
    rate. See for this particular formulation:

    D. Luong, L. Jiang, J. Kim, N. Lütkenhaus; Appl. Phys. B 122, 96 (2016)
    arXiv:1508.02811 [quant-ph]

    Optionally also returns the standard deviation of the key rate
    calculated by using propagation of error.

    Parameters
    ----------
    correlations_z : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in z-direction coincide for each raw bit.
    correlations_x : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in x-direction coincide for each raw bit.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies
    resource_list : list of scalar
        A list containing the number of resources each set of raw bits consumed.
        This might not make sense if the number of consumed resources is not
        directly assignable to one particular set of raw bits.
    return_std : bool
        Whether to also compute and return the standard deviation of the
        key rate. Default: False

    Returns
    -------
    scalar or tuple of scalars
        If return_std is False, returns the key rate.
        If return_std is True, returns at tuple of key rate and
        standard deviation of the key rate.

    """
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
            + err_corr_ineff**2
            * (-np.log2(e_z) + np.log2(1 - e_z)) ** 2
            * np.std(correlations_z) ** 2
        )
    return keyrate, keyrate_std


def calculate_keyrate_channel_use_from_time(
    correlations_z,
    correlations_x,
    err_corr_ineff,
    time_list,
    trial_time,
    return_std=False,
):
    """Calculate the asymptotic key rate per resource with only timing info.

    In some setups the number of channel uses is directly tied to time.
    CARFUL: This measure only makes sense if this is the case for the setup
    you are analyzing.

    This function calculates time intervals and resources from the `time_list`
    and then passes them to calculate_keyrate_channel_use.

    Parameters
    ----------
    correlations_z : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in z-direction coincide for each raw bit.
    correlations_x : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in x-direction coincide for each raw bit.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies
    time_list : list of scalar
        A list containing the point in time that a each set of raw bits was
        recorded. e.g. data["time"] from the data attribute of a
        requsim.tools.TwoLinkProtocol
        CAREFUL: This assumes that the protocol was started at time 0 and that
        the communication time is half the trial time.
    trial_time : scalar
        The time one trial to establish a pair takes. Usually something like
        preparation time + 2 * distance / communication speed.
    return_std : bool
        Whether to also compute and return the standard deviation of the
        key rate. Default: False

    Returns
    -------
    scalar or tuple of scalars
        If return_std is False, returns the key rate.
        If return_std is True, returns at tuple of key rate and
        standard deviation of the key rate.

    """
    time_interval_list = np.diff(pd.concat([pd.Series([trial_time / 2]), time_list]))
    resource_list = time_interval_list / trial_time
    return calculate_keyrate_channel_use(
        correlations_z=correlations_z,
        correlations_x=correlations_x,
        err_corr_ineff=err_corr_ineff,
        resource_list=resource_list,
        return_std=return_std,
    )


def standard_bipartite_evaluation(data_frame, err_corr_ineff=1):
    """Calculate fidelities and key rates from times and states.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A pandas DataFrame with columns "time" and "state", representing
        when each connection was made and the two-qubit state associated with
        that connection.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies. Default: 1

    Returns
    -------
    list of scalars
        contains: average fidelity,
                  standard deviation of fidelity,
                  average asymptotic key rate per time,
                  standard deviatoin of key rate per time

    """
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
    return [
        fidelity,
        fidelity_std,
        key_per_time,
        key_per_time_std,
    ]
