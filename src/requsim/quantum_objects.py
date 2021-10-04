import sys
import abc
from warnings import warn
from .libs.aux_functions import apply_single_qubit_map
from . import events
from collections import defaultdict
from .noise import NoiseModel

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {})


class WorldObject(ABC):
    """Abstract base class for objects that exist within a World.

    This ensures that all WorldObjects are known by the associated World and
    that they have easy access via properties to the world and its associated
    event_queue.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    label : str or None
        A string to represent this object in a human-readable way.
        If None, a default string will be used. Default: None

    Attributes
    ----------
    world : World
    label : str
        A human-readable label for the object.
    event_queue : EventQueue
    last_updated : scalar
    required_by_events : list of Events
    is_blocked : bool
    type : str

    """

    def __init__(self, world, label=None):
        self.world = world
        if label is None:
            self.label = self.world.register_world_object(self)
        else:
            self.label = label
            self.world.register_world_object(self)
        self.last_updated = self.event_queue.current_time
        self.required_by_events = []
        self.is_blocked = False

    def __str__(self):
        return self.label

    def destroy(self):
        """Remove this WorldObject from the world."""
        # in the future it might be nice to also remove associated events etc.
        self.world.deregister_world_object(self)

    @property
    def type(self):
        """Returns the quantum object type.

        Returns
        -------
        str
            The quantum object type.

        """
        return self.__class__.__name__

    @property
    def event_queue(self):
        """Shortcut to access the event_queue `self.world.event_queue`.

        Returns
        -------
        EventQueue
            The event queue

        """
        return self.world.event_queue

    def _on_update_time(self):
        pass

    def update_time(self):  # to be used to update internal time
        self._on_update_time()
        self.last_updated = self.event_queue.current_time


class Qubit(WorldObject):
    """A Qubit.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    station : Station
        The station at which the qubit is located.
    unresolved_noise : NoiseChannel or None
        Noise that affected the qubit, but has not been applied to the state
        yet. None means nothing is unresolved. Default: None
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    pair : Pair or None
        Pair if the qubit is part of a Pair, None else.
    station : Station
        The station at which the qubit is located.
    type : str
        "Qubit"
    unresolved_noise : NoiseChannel or None

    """

    # station should also know about which qubits are at its location
    def __init__(self, world, station, unresolved_noise=None, label=None):
        self.station = station
        self.unresolved_noise = unresolved_noise
        super(Qubit, self).__init__(world=world, label=label)
        self.pair = None

    def __str__(self):
        return f"{self.label} at station {self.station.label if self.station else self.station}, part of pair {self.pair.label if self.pair else self.pair}."

    @property
    def type(self):
        return "Qubit"

    def destroy(self):
        # station needs to be notified that qubit is no longer there, not sure how to handle pairs yet
        self.station.remove_qubit(self)
        super(Qubit, self).destroy()


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


class Station(WorldObject):
    """A repeater station.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    memory_noise : callable or None
        Should take parameters rho (density matrix) and t (time). Default: None
    memory_cutoff_time : scalar or None
        Qubits will be discarded after this amount of time in memory.
        Default: None
    BSM_noise_model : NoiseModel
        Noise model that is used for Bell State measurements performed at this
        station (especially for entanglement swapping).
        Default: dummy NoiseModel that corresponds to no noise.
    creation_noise_channel : NoiseChannel or None
        Noise channel that is applied to a qubit on creation. (e.g. misalignment)
        Default: None
    dark_count_probability : scalar
        Probability that a detector clicks without a state arriving.
        This is not used by the Station itself, but state generation functions
        may use this. Default: 0
    id : int or None
        [DEPRECATED: Do not use!] Label for the station. Default: None
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    qubits : list of Qubit objects
        The qubits currently at this position.
    type : str
        "Station"
    memory_noise : callable or None
    memory_cutoff_time : callable or None
    resource_tracking : defaultdict
        Intermediate store for carrying over resources used by discarded
        pairs/qubits.
    BSM_noise_model : NoiseModel
    creation_noise_channel : NoiseChannel or None
    dark_count_probability : scalar
    id : int or None

    """

    def __init__(
        self,
        world,
        position,
        memory_noise=None,
        memory_cutoff_time=None,
        BSM_noise_model=NoiseModel(),
        creation_noise_channel=None,
        dark_count_probability=0,
        id=None,
        label=None,
    ):
        self.id = id
        self.position = position
        self.qubits = []
        self.resource_tracking = defaultdict(
            lambda: {"resource_cost_add": 0, "resource_cost_max": 0}
        )
        self.memory_noise = memory_noise
        self.memory_cutoff_time = memory_cutoff_time
        self.BSM_noise_model = BSM_noise_model
        self.creation_noise_channel = creation_noise_channel
        self.dark_count_probability = dark_count_probability
        super(Station, self).__init__(world=world, label=label)

    def __str__(self):
        return f"{self.label} at position {self.position}."

    @property
    def type(self):
        return "Station"

    def create_qubit(self):
        """Create a new qubit at this station.

        Returns
        -------
        Qubit
            The created Qubit object.

        """
        new_qubit = Qubit(
            world=self.world, station=self, unresolved_noise=self.creation_noise_channel
        )
        self.qubits += [new_qubit]
        if self.memory_cutoff_time is not None:
            discard_event = events.DiscardQubitEvent(
                time=self.event_queue.current_time + self.memory_cutoff_time,
                qubit=new_qubit,
            )
            self.event_queue.add_event(discard_event)
        return new_qubit

    def remove_qubit(self, qubit):
        try:
            self.qubits.remove(qubit)
        except ValueError:
            warn(
                "Tried to remove qubit {} from station {}, but the station was not tracking that qubit.".format(
                    repr(qubit), repr(self)
                )
            )
            # print(self.event_queue.current_time)
            # print(self.event_queue.queue)
            # print(self.world.world_objects)


class Source(WorldObject):
    """A source of entangled pairs.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The two stations the source to which the source sends the entangled
        pairs, usually the neighboring repeater stations.
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The two stations the source to which the source sends the entangled
        pairs, usually the neighboring repeater stations.
    type : str
        "Source"

    """

    def __init__(self, world, position, target_stations, label=None):
        self.position = position
        self.target_stations = target_stations
        super(Source, self).__init__(world=world, label=label)

    def __str__(self):
        return (
            f"{self.label} generating states between stations "
            + ", ".join([x.label for x in self.target_stations])
            + "."
        )

    @property
    def type(self):
        return "Source"

    def generate_pair(
        self, initial_state, initial_cost_add=None, initial_cost_max=None
    ):
        """Generate an entangled pair.

        The Pair will be generated in the `initial_state` at the
        `self.target_stations` of the source.
        Usually called from a SourceEvent.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial density matrix of the two-qubit

        Returns
        -------
        Pair
            The newly generated Pair.

        """
        station1 = self.target_stations[0]
        station2 = self.target_stations[1]
        qubit1 = station1.create_qubit()
        qubit2 = station2.create_qubit()
        return Pair(
            world=self.world,
            qubits=[qubit1, qubit2],
            initial_state=initial_state,
            initial_cost_add=initial_cost_add,
            initial_cost_max=initial_cost_max,
        )


class SchedulingSource(Source):
    """A Source that schedules its next event according to a distribution.

    Parameters
    ----------
    see Source

    time_distribution : callable
        Used for scheduling. Should return the amount of time until the next
        SourceEvent should take place (possibly probabilistic).
    state_generation : callable
        Should return (possibly probabilistically) the density matrix of the
        pair generated by the source. Takes the source as input.
    label : str or None
        Optionally, provide a custom label.

    """

    def __init__(
        self,
        world,
        position,
        target_stations,
        time_distribution,
        state_generation,
        label=None,
    ):
        self.time_distribution = time_distribution
        self.state_generation = state_generation
        super(SchedulingSource, self).__init__(world, position, target_stations, label)

    def schedule_event(self):
        time_delay, times_tried = self.time_distribution(source=self)
        scheduled_time = self.event_queue.current_time + time_delay
        initial_state = self.state_generation(
            source=self
        )  # should accurately describe state at the scheduled time
        source_event = events.SourceEvent(
            time=scheduled_time,
            source=self,
            initial_state=initial_state,
            initial_cost_add=times_tried,
            initial_cost_max=times_tried,
        )
        self.event_queue.add_event(source_event)
        return source_event
