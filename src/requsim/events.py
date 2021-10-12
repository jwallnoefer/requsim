import sys
import abc
from abc import abstractmethod
from .libs import matrix as mat
from .libs.epp import dejmps_protocol
import numpy as np
import requsim.quantum_objects as quantum_objects
from collections import defaultdict
from warnings import warn

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {})


class Event(ABC):
    """Abstract base class for events.

    Events are scheduled in an EventQueue and resolved at a specific time.

    Parameters
    ----------
    time : scalar
        The time at which the event will be resolved.
    required_objects : list of QuantumObjects
        Event will only resolve if all of these still exist at `time`.
        Default: []
    priority : int (expected 0...39)
        prioritize events that happen at the same time according to this
        (lower number means being resolved first) Default: 20
    ignore_blocked : bool
        Whether the event should act even on blocked objects. Default: False

    Attributes
    ----------
    event_queue : EventQueue
        The event is part of this event queue.
        (None until added to an event queue.)
    type
    time
    required_objects
    priority
    ignore_blocked

    """

    def __init__(
        self,
        time,
        required_objects=[],
        priority=20,
        ignore_blocked=False,
        *args,
        **kwargs,
    ):
        self.time = time
        self.required_objects = required_objects
        for required_object in self.required_objects:
            assert required_object in required_object.world
            required_object.required_by_events += [self]
        self.priority = priority
        self.ignore_blocked = ignore_blocked
        self.event_queue = None
        self._return_dict = {"event_type": self.type, "resolve_successful": True}

    @abstractmethod
    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, *args, **kwargs)" % str(self.time)

    @property
    def type(self):
        """Returns the event type.

        Returns
        -------
        str
            The event type.

        """
        return self.__class__.__name__

    def _check_event_is_valid(self):
        objects_exist = np.all(
            [(req_object in req_object.world) for req_object in self.required_objects]
        )
        if self.ignore_blocked:
            objects_available = True
        else:
            objects_available = np.all(
                [not req_object.is_blocked for req_object in self.required_objects]
            )
            if objects_exist and not objects_available:
                warn(
                    "Event "
                    + str(self)
                    + " tried to access a blocked object, but is not allowed to do so. [Check whether `ignore_blocked` needs to be set.]"
                )
        return objects_exist and objects_available

    def _deregister_from_objects(self):
        for req_object in self.required_objects:
            req_object.required_by_events.remove(self)

    @abstractmethod
    def _main_effect(self):
        """Resolve the main effect of the event.

        Returns
        -------
        None or dict
            dict may optionally be used to pass information to the protocol.
            The protocol will not necessarily use this information.

        """
        pass

    def resolve(self):
        """Resolve the event.

        Returns
        -------
        dict
            dict may contain additional information that can be passed to the
            protocol. The protocol will not necessarily use this information.

        """
        if self._check_event_is_valid():
            main_return_dict = self._main_effect()
            try:
                self._return_dict.update(main_return_dict)
            except TypeError:
                # expected to happen when self._main_effect() returns None
                pass
        else:
            self._return_dict.update({"resolve_successful": False})
        self._deregister_from_objects()
        return self._return_dict


class GenericEvent(Event):
    """Event that executes arbitrary function.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    resolve_function : callable
        Function that will be called when the resolve method is called.
    *args : any
        args for resolve_function.
    required_objects : list of QuantumObjects
        Keyword only argument. Default: []
    priority : int
        Keyword only argument. Default: 20
    ignore_blocked: bool
        Keyword only argument. Default: False
    **kwargs : any
        kwargs for resolve_function.

    """

    def __init__(
        self,
        time,
        resolve_function,
        *args,
        required_objects=[],
        priority=20,
        ignore_blocked=False,
        **kwargs,
    ):
        self._resolve_function = resolve_function
        self._resolve_function_args = args
        self._resolve_function_kwargs = kwargs
        super(GenericEvent, self).__init__(
            time=time,
            required_objects=required_objects,
            priority=priority,
            ignore_blocked=ignore_blocked,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(time="
            + str(self.time)
            + ", resolve_function="
            + str(self._resolve_function)
            + ", "
            + ", ".join(map(str, self._resolve_function_args))
            + ", ".join(
                [
                    "{}={}".format(str(k), str(v))
                    for k, v in self._resolve_function_kwargs.items()
                ]
            )
            + ")"
        )

    def _main_effect(self):
        return self._resolve_function(
            *self._resolve_function_args, **self._resolve_function_kwargs
        )


class SourceEvent(Event):
    """An Event generating an entangled pair.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    source : Source
        The source object generating the entangled pair.
    initial_state : np.ndarray
        Density matrix of the two qubit system being generated.
    *args, **kwargs :
        additional optional args and kwargs to pass to the the
        generate_pair method of `source`

    Attributes
    ----------
    source
    initial_state
    generation_args : additional args for the generate_pair method of source
    generation_kwargs : additional kwargs for the generate_pair method of source

    """

    def __init__(self, time, source, initial_state, *args, **kwargs):
        self.source = source
        self.initial_state = initial_state
        self.generation_args = args
        self.generation_kwargs = kwargs
        super(SourceEvent, self).__init__(
            time=time, required_objects=[self.source, *self.source.target_stations]
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(time={}, source={}, initial_state={})".format(
                str(self.time), str(self.source), repr(self.initial_state)
            )
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} at time={self.time} generating a state between stations "
            + ", ".join([x.label for x in self.source.target_stations])
            + "."
        )

    def _main_effect(self):
        """Resolve the event.

        Generates a pair at the target stations of `self.source`.

        Returns
        -------
        None

        """
        # print("A source event happened at time", self.time, "while queue looked like this:", self.event_queue.queue)
        self.source.generate_pair(
            self.initial_state, *self.generation_args, **self.generation_kwargs
        )


class EntanglementSwappingEvent(Event):
    """An event to perform entanglement swapping.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    pairs : list of Pairs
        The left pair and the right pair.
    error_func : callable or None [Deprecated, use station.BSM_noise_model instead.]
        A four-qubit map. Careful: This overwrites any noise behavior set by
        station. Default: None

    Attributes
    ----------
    pairs
    error_func

    """

    def __init__(self, time, pairs, error_func=None):
        self.pairs = pairs
        self.error_func = error_func  # currently a four-qubit channel, would be nicer as two-qubit channel that gets applied to the right qubits
        super(EntanglementSwappingEvent, self).__init__(
            time=time,
            required_objects=self.pairs
            + [qubit for pair in self.pairs for qubit in pair.qubits],
        )

    def __repr__(self):
        return self.__class__.__name__ + "(time={}, pairs={}, error_func={})".format(
            str(self.time), str(self.pairs), repr(self.error_func)
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} at time={self.time} using pairs "
            + ", ".join([x.label for x in self.pairs])
            + "."
        )

    def _main_effect(self):
        """Resolve the event.

        Performs entanglement swapping between the two pairs and generates the
        appropriate Pair object for the long-distance pair.

        Returns
        -------
        None

        """
        # it would be nice if this could handle arbitrary configurations
        # instead of relying on strict indexes of left and right pairs
        left_pair = self.pairs[0]
        right_pair = self.pairs[1]
        assert left_pair.qubits[1].station is right_pair.qubits[0].station
        swapping_station = left_pair.qubits[1].station
        left_pair.update_time()
        right_pair.update_time()
        four_qubit_state = mat.tensor(left_pair.state, right_pair.state)
        # non-ideal-bell-measurement
        if self.error_func is not None:
            four_qubit_state = self.error_func(four_qubit_state)
        elif swapping_station.BSM_noise_model.channel_before is not None:
            noise_channel = swapping_station.BSM_noise_model.channel_before
            if noise_channel.n_qubits == 4:
                four_qubit_state = noise_channel(four_qubit_state)
            elif noise_channel.n_qubits == 2:
                four_qubit_state = noise_channel.apply_to(
                    rho=four_qubit_state, qubit_indices=[1, 2]
                )
            else:
                raise ValueError(
                    "Error Channel: "
                    + str(noise_channel)
                    + " is not supported by Bell State Measurement. Expects a 2- or 4-qubit channel."
                )
        if swapping_station.BSM_noise_model.map_replace is not None:
            two_qubit_state = swapping_station.BSM_noise_model.map_replace(
                four_qubit_state
            )
        else:  # do the main thing
            my_proj = mat.tensor(mat.I(2), mat.phiplus, mat.I(2))
            two_qubit_state = np.dot(np.dot(mat.H(my_proj), four_qubit_state), my_proj)
            two_qubit_state = two_qubit_state / np.trace(two_qubit_state)
        if (
            swapping_station.BSM_noise_model.channel_after is not None
        ):  # not sure this even makes sense in this context because qubits at station are expected to be gone
            noise_channel = swapping_station.BSM_noise_model.channel_after
            assert noise_channel.n_qubits == 2
            two_qubit_state = noise_channel(two_qubit_state)
        if (
            left_pair.resource_cost_add is not None
            and right_pair.resource_cost_add is not None
        ):
            new_cost_add = left_pair.resource_cost_add + right_pair.resource_cost_add
        else:
            new_cost_add = None
        if (
            left_pair.resource_cost_max is not None
            and right_pair.resource_cost_max is not None
        ):
            new_cost_max = max(
                left_pair.resource_cost_max, right_pair.resource_cost_max
            )
        else:
            new_cost_max = None
        new_pair = quantum_objects.Pair(
            world=left_pair.world,
            qubits=[left_pair.qubits[0], right_pair.qubits[1]],
            initial_state=two_qubit_state,
            initial_cost_add=new_cost_add,
            initial_cost_max=new_cost_max,
        )
        # cleanup
        left_pair.qubits[1].destroy()
        right_pair.qubits[0].destroy()
        left_pair.destroy()
        right_pair.destroy()


class DiscardQubitEvent(Event):
    """Event to discard a qubit and associated pair.

    For example if the qubit sat in memory too long and is discarded.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    qubit : Qubit
        The Qubit that will be discarded.
    priority : int
        Default: 39 (because discard events should get processed last)
    ignore_blocked : bool
        Whether the event should act on blocked quantum objects. Default: True

    Attributes
    ----------
    qubit

    """

    def __init__(self, time, qubit, priority=39, ignore_blocked=True):
        self.qubit = qubit
        super(DiscardQubitEvent, self).__init__(
            time=time,
            required_objects=[self.qubit],
            priority=priority,
            ignore_blocked=True,
        )

    def __repr__(self):
        return self.__class__.__name__ + "(time={}, qubit={})".format(
            str(self.time), str(self.qubit)
        )

    def __str__(self):
        return f"{self.__class__.__name__} at time={self.time} to discard {self.qubit.label}."

    def _main_effect(self):
        """Discards the qubit and associated pair, if the qubit still exists.

        Returns
        -------
        None

        """
        if self.qubit.pair is not None:
            self.qubit.pair.destroy_and_track_resources()
            self.qubit.pair.qubits[0].destroy()
            self.qubit.pair.qubits[1].destroy()
        else:
            self.qubit.destroy()
            # print("A Discard Event happened with eventqueue:", self.qubit.world.event_queue.queue)


class EntanglementPurificationEvent(Event):
    """Short summary.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    pairs : list of Pairs
        The pairs involved in the entanglement purification protocol.
    protocol : {"dejmps"} or callable
        Can be one of the pre-installed or an arbitrary callable that takes
        a tensor product of pair states as input and returns a tuple of
        (success probability, state of a single pair) back.
        So far only supports n->1 protocols.
    communication_speed : scalar
        speed at which the classical information travels
        Default: 2*10^8 (speed of light in optical fibre)


    Attributes
    ----------
    pairs
    protocol
    communication_speed

    """

    def __init__(self, time, pairs, protocol="dejmps", communication_speed=2e8):
        self.pairs = pairs
        if protocol == "dejmps":
            self.protocol = dejmps_protocol
        elif callable(protocol):
            self.protocol = protocol
        else:
            raise ValueError(
                "EntanglementPurificationEvent got a protocol type that is not supported: "
                + repr(protocol)
            )
        self.communication_speed = communication_speed
        super(EntanglementPurificationEvent, self).__init__(
            time=time,
            required_objects=self.pairs
            + [qubit for pair in self.pairs for qubit in pair.qubits],
        )

    def __repr__(self):
        return self.__class__.__name__ + "(time={}, pairs={}, protocol={})".format(
            repr(self.time), repr(self.pairs), repr(self.protocol)
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} at time={self.time} using pairs "
            + ", ".join([x.label for x in self.pairs])
            + f" with protocol {self.protocol}."
        )

    def _main_effect(self):
        """Probabilistically performs the entanglement purification protocol.

        Returns
        -------
        None

        """
        # probably could use a check that pairs are between same stations?
        for pair in self.pairs:
            pair.update_time()
        rho = mat.tensor(*[pair.state for pair in self.pairs])
        p_suc, state = self.protocol(rho)
        output_pair = self.pairs[0]
        output_pair.state = state
        if output_pair.resource_cost_add is not None:
            output_pair.resource_cost_add = np.sum(
                [pair.resource_cost_add for pair in self.pairs]
            )
        if output_pair.resource_cost_max is not None:
            output_pair.resource_cost_max = np.sum(
                [pair.resource_cost_max for pair in self.pairs]
            )
        output_pair.is_blocked = True
        output_pair.qubit1.is_blocked = True
        output_pair.qubit2.is_blocked = True
        for pair in self.pairs[1:]:  # pairs that have been destroyed in the process
            pair.qubits[0].destroy()
            pair.qubits[1].destroy()
            pair.destroy()
        communication_time = (
            np.abs(
                output_pair.qubit2.station.position
                - output_pair.qubit1.station.position
            )
            / self.communication_speed
        )
        if np.random.random() <= p_suc:  # if successful
            unblock_event = UnblockEvent(
                time=self.time + communication_time,
                quantum_objects=[output_pair, output_pair.qubit1, output_pair.qubit2],
            )
            self.event_queue.add_event(unblock_event)
            return {"output_pair": output_pair, "is_successful": True}
        else:  # if unsuccessful

            def destroy_function():
                output_pair.destroy_and_track_resources()
                output_pair.qubits[0].destroy()
                output_pair.qubits[1].destroy()

            destroy_event = GenericEvent(
                time=self.time + communication_time,
                resolve_function=destroy_function,
                required_objects=[output_pair],
                priority=0,
                ignore_blocked=True,
            )
            self.event_queue.add_event(destroy_event)
            return {"output_pair": output_pair, "is_successful": False}


class UnblockEvent(Event):
    """Unblock a set of quantum objects.

    This is useful to mark the time when necessary classical information has
    arrived and the quantum objects may be used by the protocol again.
    (e.g. after entanglement purification)

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    quantum_objects : list of QuantumObjects
        The quantum objects to be unblocked.
    priority : int (expected 0...39)
        Default: 0 (because unblocking should happen as soon as possible)

    Attributes
    ----------
    quantum_objects

    """

    def __init__(self, time, quantum_objects, priority=0):
        self.quantum_objects = quantum_objects
        super(UnblockEvent, self).__init__(
            time=time,
            required_objects=self.quantum_objects,
            priority=priority,
            ignore_blocked=True,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(time={}, quantum_objects={}, priority={})".format(
                repr(self.time), repr(self.quantum_objects), repr(self.priority)
            )
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} at time={self.time} for objects "
            + ", ".join([x.label for x in self.quantum_objects])
            + "."
        )

    def _main_effect(self):
        for quantum_object in self.quantum_objects:
            quantum_object.is_blocked = False
        return {"unblocked_objects": self.quantum_objects}


class EventQueue(object):
    """Provides methods to queue and resolve Events in order.

    Attributes
    ----------
    queue : list of Events
        An ordered list of future events to resolve.
    current_time : scalar
        The current time of the event queue.

    """

    def __init__(self):
        self.queue = []
        self.current_time = 0
        self._stats = defaultdict(
            lambda: {"scheduled": 0, "resolved": 0, "resolved_successfully": 0}
        )

    def __str__(self):
        return "EventQueue: " + str(self.queue)

    def __len__(self):
        return len(self.queue)

    @property
    def next_event(self):
        """Helper property to access next scheduled event.

        Returns
        -------
        Event or None
            The next scheduled event. None if the event queue is empty.

        """
        try:
            return self.queue[0]
        except IndexError:
            return None

    def _insert_event(self, event):
        """Insert event at appropriate position in the queue.

        This uses bisection instead of sort to avoid needlessly constructing
        (x.time, x.priority) for all events in the queue as would be done by
        list.sort()
        Insertion is done to the right of matching events.

        Parameters
        ----------
        event : Event
            The event that will be inserted.

        Returns
        -------
        None

        """
        lo = 0
        hi = len(self.queue)
        x = (event.time, event.priority)
        while lo < hi:
            mid = (lo + hi) // 2
            mid_event = self.queue[mid]
            if x < (mid_event.time, mid_event.priority):
                hi = mid
            else:
                lo = mid + 1
        self.queue.insert(lo, event)

    def add_event(self, event):
        """Add an event to the queue.

        The queue is sorted again in order to schedule the event at the correct
        time.

        Parameters
        ----------
        event : Event
            The Event to be added to the queue.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `event.time` is in the past.

        """
        if event.time < self.current_time:
            raise ValueError(
                "EventQueue.add_event tried to schedule an event in the past."
            )
        event.event_queue = self
        self._insert_event(event)
        self._stats[event.type]["scheduled"] += 1

    def resolve_next_event(self):
        """Remove the next scheduled event from the queue and resolve it.

        Returns
        -------
        dict:
            A dict with at least keys "event_type" and "resolve_successful",
            may have additional keys with additional information that can be
            passed to the protocol.

        """
        event = self.queue[0]
        self.current_time = event.time
        return_message = event.resolve()
        self.queue = self.queue[1:]
        self._stats[event.type]["resolved"] += 1
        if return_message["resolve_successful"]:
            self._stats[event.type]["resolved_successfully"] += 1
        return return_message

    def resolve_until(self, target_time):
        """Resolve events until `target_time` is reached.

        Parameters
        ----------
        target_time : scalar
            Resolve until current_time is this.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `target_time` lies in the past.
        """
        if target_time < self.current_time:
            raise ValueError(
                "EventQueue.resolve_until cannot resolve to a time in the past."
            )
        while self.queue:
            event = self.queue[0]
            if event.time <= target_time:
                self.resolve_next_event()
            else:
                break
        self.current_time = target_time

    def advance_time(self, time_interval):
        """Helper method to manually advance time.

        Parameters
        ----------
        time_interval : int
            The amount of time that passes.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an event is skipped during the `time_interval`.
        """
        self.current_time += time_interval
        if self.queue and self.queue[0].time < self.current_time:
            raise ValueError(
                "time_interval too large. Manual time advancing skipped an event. Time travel is not permitted."
            )

    def print_stats(self):
        """Print stats about the events scheduled and resolved so far.

        Returns
        -------
        None

        """
        for event_type, count_dict in self._stats.items():
            string_parts = [
                f"{event_type}:",
                f"{count_dict['scheduled']} scheduled",
                f"{count_dict['resolved']} resolved",
                f"{count_dict['resolved_successfully']} resolved successfully",
            ]
            print("\n    ".join(string_parts))
