from . import WorldObject
from ..libs.aux_functions import apply_single_qubit_map


class Pair(WorldObject):
    """A Pair of two qubits with its associated quantum state.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    qubits : list of Qubits
        The two qubits that are part of this entangled Pair.
    initial_state : np.ndarray
        The two qubit system is intialized with this density matrix.
    initial_cost_add : scalar or None
        Initial resource cost (in cumulative channel uses). Can be left None if
        tracking is not done. Default: None
    initial_cost_max : scalar or None
        Initial resource cost (in max channel uses). Can be left None if
        tracking is not done. Default: None
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    state : np.ndarray
        Current density matrix of this two qubit system.
    qubit1 : Qubit
        Alternative way to access `self.qubits[0]`
    qubit2 : Qubit
        Alternative way to access `self.qubits[1]`
    qubits : List of qubits
        The two qubits that are part of this entangled Pair.
    resource_cost_add : scalar or None
        cumulative channel uses that were needed to create this pair.
        None means resource are not tracked.
    resource_cost_max : scalar or None
        max channel uses that were needed to create this pair.
        None means resource are not tracked.
    type : str
        "Pair"

    """

    def __init__(
        self,
        world,
        qubits,
        initial_state,
        label=None,
    ):
        # maybe add a check that qubits are always in the same order?
        self.qubits = qubits
        self.state = initial_state
        self.qubit1.update_info({"pair": self})
        self.qubit1.add_destroy_callback(self._on_qubit_destroy)
        self.qubit2.update_info({"pair": self})
        self.qubit2.add_destroy_callback(self._on_qubit_destroy)
        # add self as noise handler for its qubits
        self.qubit1.add_noise_handler(self._qubit1_noise_handler)
        self.qubit2.add_noise_handler(self._qubit2_noise_handler)
        super(Pair, self).__init__(world=world, label=label)

    def __str__(self):
        return (
            f"{self.label} with qubits "
            + ", ".join([x.label for x in self.qubits])
            + " between stations "
            + ", ".join(
                [x.station.label if x.station else str(x.station) for x in self.qubits]
            )
            + "."
        )

    @property
    def type(self):
        return "Pair"

    # not sure we actually need to be able to change qubits
    @property
    def qubit1(self):
        """Alternative way to access `self.qubits[0]`.

        Returns
        -------
        Qubit
            The first qubit of the pair.

        """
        return self.qubits[0]

    @qubit1.setter
    def qubit1(self, qubit):
        self.qubits[0] = qubit

    @property
    def qubit2(self):
        """Alternative way to access `self.qubits[1]`.

        Returns
        -------
        Qubit
            The second qubit of the pair.

        """
        return self.qubits[1]

    @qubit2.setter
    def qubit2(self, qubit):
        self.qubits[1] = qubit

    def _qubit1_noise_handler(self, noise_channel, *args, **kwargs):
        self.state = noise_channel.apply_to(
            rho=self.state, qubit_indices=[0], *args, **kwargs
        )

    def _qubit2_noise_handler(self, noise_channel, *args, **kwargs):
        self.state = noise_channel.apply_to(
            rho=self.state, qubit_indices=[1], *args, **kwargs
        )

    def _on_qubit_destroy(self, qubit):
        if qubit in self.qubits:
            self.destroy()

    def is_between_stations(self, station1, station2):
        return (
            self.qubit1.station == station1 and self.qubit2.station == station2
        ) or (self.qubit1.station == station2 and self.qubit2.station == station1)

    def _on_update_time(self):
        self.qubit1.update_time()
        self.qubit2.update_time()

    def destroy(self):
        # remove self as noise handler for its qubits
        self.qubit1.remove_noise_handler(self._qubit1_noise_handler)
        self.qubit2.remove_noise_handler(self._qubit2_noise_handler)
        super(Pair, self).destroy()
