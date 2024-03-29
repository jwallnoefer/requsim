from . import WorldObject


class MultiQubit(WorldObject):
    """A quantum object representing a state of multiple qubits.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    qubits : list of Qubits
        Multiple qubits that are part of this collection.
    initial_state : np.ndarray
        The multi-qubit system is initialized with this density matrix.
        Shape should fit the number of qubits in `qubits`.
    label : str or None, optional
        Optionally, provide a custom label.

    Attributes
    ----------
    state : np.ndarray
        Current density matrix of this n-qubit system.

    """

    def __init__(self, world, qubits, initial_state, label=None):
        self._qubits = tuple(qubits)
        self._num_qubits = len(self._qubits)
        self.state = initial_state
        self._noise_handler_by_qubit = {}
        for qubit_index, qubit in enumerate(self._qubits):
            qubit.update_info({"higher_order_object": self})
            qubit.higher_order_object = self
            qubit.add_destroy_callback(self._on_qubit_destroy)
            noise_handler = self._generate_qubit_noise_handler(qubit_index)
            self._noise_handler_by_qubit[qubit] = noise_handler
            qubit.add_noise_handler(noise_handler)
        super(MultiQubit, self).__init__(world=world, label=label)

    def __repr__(self):
        return (
            f'<{self.__class__.__module__}.{self.__class__.__name__} "{self.label}" '
            + "qubits=["
            + ", ".join([x.label for x in self.qubits])
            + "]"
            + f" with state at last update {self.state}"
            + ">"
        )

    def __str__(self):
        return (
            f"{self.label} with qubits"
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
        return f"{self._num_qubits}-qubit MultiQubit"

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def qubits(self):
        return tuple(self._qubits)

    def _on_update_time(self):
        for qubit in self._qubits:
            qubit.update_time()

    def _on_qubit_destroy(self, qubit):
        if qubit in self._qubits:
            self.destroy()

    def _generate_qubit_noise_handler(self, qubit_index):
        def qubit_noise_handler(noise_channel, *args, **kwargs):
            self.state = noise_channel.apply_to(
                rho=self.state, qubit_indices=[qubit_index], *args, **kwargs
            )
            handling_successful = True
            return handling_successful

        return qubit_noise_handler

    def destroy(self):
        for qubit in self._qubits:
            if qubit in qubit.world:  # doesn't need to get deleted twice
                qubit.remove_noise_handler(self._noise_handler_by_qubit[qubit])
        super(MultiQubit, self).destroy()
