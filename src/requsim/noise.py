import sys
from .libs.aux_functions import apply_single_qubit_map, apply_m_qubit_map


class NoiseChannel(object):
    """Standardized way to define noise channels.

    This class can be simply called to apply this channel on a state of the
    correct number of qubits, or you can use the apply_to method to let the
    noise channel handle its application for you.

    Parameters
    ----------
    n_qubits : int
        This noise channel acts on `n_qubits` input qubits.
    channel_function : callable
        The function describing the channel.

    Attributes
    ----------
    n_qubits

    """

    def __init__(self, n_qubits, channel_function):
        self.n_qubits = n_qubits
        self._channel_func = channel_function

    def __call__(self, rho, *args, **kwargs):
        return self._channel_func(rho, *args, **kwargs)

    def apply_to(self, rho, qubit_indices, *args, **kwargs):
        # additional functionality e.g. for different ways to define channels may be added here
        """Apply the noise channel to specified qubits of a state `rho`.

        Parameters
        ----------
        rho : np.ndarray
            The density matrix the channel is applied to.
        qubit_indices : list of ints
            Numbering of qubits runs from 0...n-1
        *args, **kwargs : are handed through to the appropriate function

        Returns
        -------
        np.ndarray
            The state after the channel has been applied.

        """
        assert len(qubit_indices) == self.n_qubits
        if self.n_qubits == 1:
            return apply_single_qubit_map(
                map_func=self, qubit_index=qubit_indices[0], rho=rho, *args, **kwargs
            )
        else:
            return apply_m_qubit_map(
                map_func=self, qubit_indices=qubit_indices, rho=rho, *args, **kwargs
            )


class NoiseModel(object):
    """Class for describing noise in a standardized way.

    Parameters
    ----------
    channel_before : NoiseChannel or None
        Channel that is applied before the operation. Default: None
    map_replace : callable or None
        This noisy callable replaces whatever the operation tries to do.
        Default: None (May not be supported by all processes.)
    channel_after : NoiseChannel or None
        Channel that is applied after the operation. Default: None

    Attributes
    ----------
    channel_before
    map_replace
    channel_after

    """

    def __init__(self, channel_before=None, map_replace=None, channel_after=None):
        self.channel_before = channel_before
        self.map_replace = map_replace
        self.channel_after = channel_after
