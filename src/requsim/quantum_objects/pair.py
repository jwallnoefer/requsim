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
        initial_cost_add=None,
        initial_cost_max=None,
        label=None,
    ):
        # maybe add a check that qubits are always in the same order?
        self.qubits = qubits
        self.state = initial_state
        self.qubit1.pair = self
        self.qubit2.pair = self
        self.resource_cost_add = initial_cost_add
        self.resource_cost_max = initial_cost_max
        # if there are lingering resources trackings to be done, add them now
        if self.resource_cost_add is not None or self.resource_cost_max is not None:
            resources1 = self.qubit1.station.resource_tracking[self.qubit2.station]
            resources2 = self.qubit2.station.resource_tracking[self.qubit1.station]
            assert resources1 == resources2
            if self.resource_cost_add is not None:
                self.resource_cost_add += resources1["resource_cost_add"]
                # then reset count
                resources1[
                    "resource_cost_add"
                ] = 0  # changing the mutable object will also change it in the real tracking dictionary
                resources2["resource_cost_add"] = 0
            if self.resource_cost_max is not None:
                self.resource_cost_max += resources1["resource_cost_max"]
                # then reset count
                resources1[
                    "resource_cost_max"
                ] = 0  # changing the mutable object will also change it in the real tracking dictionary
                resources2["resource_cost_max"] = 0
        # apply unresolved channels of the qubits
        if self.qubit1.unresolved_noise is not None:
            self.state = self.qubit1.unresolved_noise.apply_to(
                rho=self.state, qubit_indices=[0]
            )
            self.qubit1.unresolved_noise = None
        if self.qubit2.unresolved_noise is not None:
            self.state = self.qubit2.unresolved_noise.apply_to(
                rho=self.state, qubit_indices=[1]
            )
            self.qubit2.unresolved_noise = None

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

    def is_between_stations(self, station1, station2):
        return (
            self.qubit1.station == station1 and self.qubit2.station == station2
        ) or (self.qubit1.station == station2 and self.qubit2.station == station1)

    def _on_update_time(self):
        time_interval = self.event_queue.current_time - self.last_updated
        map0 = self.qubits[0].station.memory_noise
        if map0 is not None:
            self.state = apply_single_qubit_map(
                map_func=map0, qubit_index=0, rho=self.state, t=time_interval
            )
        map1 = self.qubits[1].station.memory_noise
        if map1 is not None:
            self.state = apply_single_qubit_map(
                map_func=map1, qubit_index=1, rho=self.state, t=time_interval
            )

    def destroy_and_track_resources(self):
        station1 = self.qubits[0].station
        station2 = self.qubits[1].station
        if self.resource_cost_add is not None:
            station1.resource_tracking[station2][
                "resource_cost_add"
            ] += self.resource_cost_add
            station2.resource_tracking[station1][
                "resource_cost_add"
            ] += self.resource_cost_add
        if self.resource_cost_max is not None:
            station1.resource_tracking[station2][
                "resource_cost_max"
            ] += self.resource_cost_max
            station2.resource_tracking[station1][
                "resource_cost_max"
            ] += self.resource_cost_max
        self.destroy()
