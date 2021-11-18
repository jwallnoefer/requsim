import numpy as np


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
            raise ValueError(
                f"Can't calculate distance between positions with shape {pos1.shape} and {pos2.shape}"
            )
    else:
        raise TypeError(
            f"Can't calculate distance between positions of type {type(pos1)} and type {type(pos2)}"
        )


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
