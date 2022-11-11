from . import WorldObject


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
    label : str or None, optional
        Optionally, provide a custom label.

    Attributes
    ----------
    state : np.ndarray
        Current density matrix of this two qubit system.

    """

    def __init__(
        self,
        world,
        qubits,
        initial_state,
        label=None,
    ):
        # maybe add a check that qubits are always in the same order?
        self._qubits = tuple(qubits)
        self.state = initial_state
        self.qubit1.update_info({"pair": self})
        self.qubit1.higher_order_object = self
        self.qubit1.add_destroy_callback(self._on_qubit_destroy)
        self.qubit2.update_info({"pair": self})
        self.qubit2.higher_order_object = self
        self.qubit2.add_destroy_callback(self._on_qubit_destroy)
        # add self as noise handler for its qubits
        self.qubit1.add_noise_handler(self._qubit1_noise_handler)
        self.qubit2.add_noise_handler(self._qubit2_noise_handler)
        super(Pair, self).__init__(world=world, label=label)

    def __repr__(self):
        return self.__class__.__name__ + (
            f"(world={self.world}, qubits={self.qubits}, "
            f"initial_state={self.state}, label={self.label})"
        )

    def __str__(self):
        return (
            f"{self.label} with qubits "
            + ", ".join([x.label for x in self.qubits])
            + " between stations "
            + ", ".join(
                [
                    x._info["station"].label
                    if x._info["station"]
                    else str(x._info["station"])
                    for x in self.qubits
                ]
            )
            + "."
        )

    @property
    def type(self):
        return "Pair"

    @property
    def qubits(self):
        return tuple(self._qubits)

    # not sure we actually need to be able to change qubits
    @property
    def qubit1(self):
        """Alternative way to access `self.qubits[0]`.

        Returns
        -------
        Qubit
            The first qubit of the pair.

        """
        return self._qubits[0]

    @property
    def qubit2(self):
        """Alternative way to access `self.qubits[1]`.

        Returns
        -------
        Qubit
            The second qubit of the pair.

        """
        return self._qubits[1]

    def _qubit1_noise_handler(self, noise_channel, *args, **kwargs):
        self.state = noise_channel.apply_to(
            rho=self.state, qubit_indices=[0], *args, **kwargs
        )
        handling_successful = True
        return handling_successful

    def _qubit2_noise_handler(self, noise_channel, *args, **kwargs):
        self.state = noise_channel.apply_to(
            rho=self.state, qubit_indices=[1], *args, **kwargs
        )
        handling_successful = True
        return handling_successful

    def _on_qubit_destroy(self, qubit):
        if qubit in self.qubits:
            self.destroy()

    def is_between_stations(self, station1, station2):
        """Check whether qubits are at specified stations.

        Parameters
        ----------
        station1 : Station
        station2 : Station

        Returns
        -------
        bool
            True if pair is between station1 and station2, False otherwise.

        """
        return (self.qubit1 in station1.qubits and self.qubit2 in station2.qubits) or (
            self.qubit1 in station2.qubits and self.qubit2 in station1.qubits
        )

    def _on_update_time(self):
        self.qubit1.update_time()
        self.qubit2.update_time()

    def destroy(self):
        # remove self as noise handler for its qubits
        if self.qubit1 in self.world:
            self.qubit1.remove_noise_handler(self._qubit1_noise_handler)
        if self.qubit2 in self.world:
            self.qubit2.remove_noise_handler(self._qubit2_noise_handler)
        super(Pair, self).destroy()
