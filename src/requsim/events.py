from abc import ABC, abstractmethod
from .libs import matrix as mat
from .libs.epp import dejmps_protocol
import numpy as np
import requsim.quantum_objects as quantum_objects
from collections import defaultdict
from warnings import warn


class Event(ABC):
    """Abstract base class for events.

    Events are scheduled in an EventQueue and resolved at a specific time.
    The resolution is performed by calling the Event.resolve method once.
    Subclasses of Event need to overwrite the Event._main_effect method
    to perform the desired action the event represents. All references
    the _main_effect needs should be provided at initialization.

    Parameters
    ----------
    time : scalar
        The time at which the event will be resolved.
    required_objects : list of QuantumObjects, or None
        Event will only resolve if all of these still exist at `time`.
        Default: None
    priority : int (expected 0...39)
        prioritize events that happen at the same time according to this
        (lower number means being resolved first) Default: 20
    ignore_blocked : bool
        Whether the event should act even on blocked objects. Default: False
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None

    Attributes
    ----------
    event_queue : EventQueue
        The event is part of this event queue.
        (None until added to an event queue.)
    time
    required_objects
    priority
    ignore_blocked
    callback_functions

    """

    def __init__(
        self,
        time,
        required_objects=None,
        priority=20,
        ignore_blocked=False,
        callback_functions=None,
        *args,
        **kwargs,
    ):
        self.time = time
        if required_objects is None:
            required_objects = []
        self.required_objects = required_objects
        for required_object in self.required_objects:
            assert required_object in required_object.world
            required_object.required_by_events += [self]
        self.priority = priority
        self.ignore_blocked = ignore_blocked
        self.event_queue = None
        self._return_dict = {
            "event": self,
            "event_type": self.type,
            "resolve_successful": True,
        }
        if callback_functions is None:
            callback_functions = []
        self._callback_functions = callback_functions

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

        This is the main method that needs to be overwritten
        by subclasses.

        Returns
        -------
        None or dict
            dict may optionally be used to pass information that
            can be used by a protocol or callbacks. Will be used
            to update the return dict of the resolve method.

        """
        pass

    def resolve(self):
        """Resolve the event.

        Returns
        -------
        dict
            The return dict dict can be used by a protocol or
            callbacks. Specific events may provide additional
            key-value pairs, but will always contain at least:

            "event" : Event
                The event object itself. While all necessary
                information should be provided with separate
                keys, this can be used as a fallback.
            "event_type" : str
                The type property of this event.
            "resolve_successful" : bool
                False if something prevented the resolution
                of the event, e.g. the required quantum
                objects no longer exist. Note that this indicates
                only whether the event could be resolved according
                to the event system rules, and not the
                success or failure of an event with an inherently
                probabilistic effect (such as entanglement purification).
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
        # callbacks after resolving
        for callback_func in self._callback_functions:
            callback_func(self._return_dict)
        return self._return_dict

    def add_callback(self, callback_func):
        """Add a callback to this event.

        Multiple callbacks added this way will resolve in the order they were
        added.

        Parameters
        ----------
        callback_func : callable
            This function will be called with the `_return_dict` as an argument
            after the event is resolved.

        Returns
        -------
        None

        """
        self._callback_functions += [callback_func]


class GenericEvent(Event):
    """Event that executes arbitrary function.

    Additional information in return dict of resolve method: whatever `resolve_function` returns.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    resolve_function : callable
        Function that will be called when the resolve method is called.
    *args : any
        args for resolve_function.
    required_objects : list of QuantumObjects, or None
        Keyword only argument. Default: None
    priority : int
        Keyword only argument. Default: 20
    ignore_blocked: bool
        Keyword only argument. Default: False
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None
    **kwargs : any
        kwargs for resolve_function.

    """

    def __init__(
        self,
        time,
        resolve_function,
        *args,
        required_objects=None,
        priority=20,
        ignore_blocked=False,
        callback_functions=None,
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
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(time={self.time}, resolve_function={self._resolve_function},"
            + ", ".join(map(str, self._resolve_function_args))
            + f"required_objects={self.required_objects}, priority={self.priority}, ignore_blocked={self.ignore_blocked}, "
            + f"callback_functions={self._callback_functions}, "
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

    Additional information in return dict of resolve method:

    "source" : Source
        The source that generated the pair.
    "output_pair" : Pair
        The pair that was generated.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    source : Source
        The source object generating the entangled pair.
    initial_state : np.ndarray
        Density matrix of the two qubit system being generated.
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None
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

    def __init__(
        self, time, source, initial_state, callback_functions=None, *args, **kwargs
    ):
        self.source = source
        self.initial_state = initial_state
        self.generation_args = args
        self.generation_kwargs = kwargs
        super(SourceEvent, self).__init__(
            time=time,
            required_objects=[self.source, *self.source.target_stations],
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(time={self.time}, source={self.source}, initial_state={self.initial_state}, "
            + f"callback_functions={self._callback_functions}, "
            + ", ".join(map(str, self.generation_args))
            + ", ".join(
                [
                    "{}={}".format(str(k), str(v))
                    for k, v in self.generation_kwargs.items()
                ]
            )
            + ")"
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
        dict
            The return_dict of this event is updated with this.

        """
        new_pair = self.source.generate_pair(
            self.initial_state, *self.generation_args, **self.generation_kwargs
        )
        return {"source": self.source, "output_pair": new_pair}


class EntanglementSwappingEvent(Event):
    """An event to perform entanglement swapping.

    Additional information in return dict of resolve method:

    "output_pair" : Pair
        The resulting pair after the entanglement swapping operation.
    "swapping_station" : Station
        The station that performed the entanglement swapping.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    pairs : list of Pairs
        The left pair and the right pair.
    station : Station
        The station where the entanglement swapping is performed.
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None

    Attributes
    ----------
    pairs
    station

    """

    def __init__(self, time, pairs, station, callback_functions=None):
        self.pairs = pairs
        self.station = station
        super(EntanglementSwappingEvent, self).__init__(
            time=time,
            required_objects=self.pairs
            + [qubit for pair in self.pairs for qubit in pair.qubits],
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(time={self.time}, pairs={self.pairs}), "
            + f"station={self.station}, "
            + f"callback_functions={self._callback_functions}"
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
        dict
            The return_dict of this event is updated with this.

        """
        pair1, pair2 = self.pairs
        # find qubits at swapping station
        swapping_qubits = []
        swapping_indices = []
        leftover_qubits = []
        leftover_indices = []
        for idx, qubit in enumerate(pair1.qubits):
            if qubit in self.station.qubits:
                swapping_qubits += [qubit]
                swapping_indices += [idx]
            else:
                leftover_qubits += [qubit]
                leftover_indices += [idx]
        assert len(swapping_qubits) == 1
        for idx, qubit in enumerate(pair2.qubits):
            if qubit in self.station.qubits:
                swapping_qubits += [qubit]
                swapping_indices += [idx + 2]
            else:
                leftover_qubits += [qubit]
                leftover_indices += [idx + 2]
        assert len(swapping_qubits) == 2
        pair1.update_time()
        pair2.update_time()
        four_qubit_state = mat.tensor(pair1.state, pair2.state)
        four_qubit_state = mat.reorder(
            rho=four_qubit_state,
            sys=[
                leftover_indices[0],
                swapping_indices[0],
                swapping_indices[1],
                leftover_indices[1],
            ],
        )
        noise_model = self.station.BSM_noise_model
        # non-ideal-bell-measurement
        if noise_model.channel_before is not None:
            noise_channel = noise_model.channel_before
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
        if noise_model.map_replace is not None:
            two_qubit_state = noise_model.map_replace(four_qubit_state)
        else:  # do the main thing
            my_proj = mat.tensor(mat.I(2), mat.phiplus, mat.I(2))
            two_qubit_state = np.dot(np.dot(mat.H(my_proj), four_qubit_state), my_proj)
            two_qubit_state = two_qubit_state / np.trace(two_qubit_state)
        if noise_model.channel_after is not None:
            # not sure this even makes sense in this context because qubits at station are expected to be gone
            noise_channel = noise_model.channel_after
            assert noise_channel.n_qubits == 2
            two_qubit_state = noise_channel(two_qubit_state)
        new_pair = quantum_objects.Pair(
            world=pair1.world,
            qubits=leftover_qubits,
            initial_state=two_qubit_state,
        )
        # cleanup
        for qubit in swapping_qubits:
            qubit.destroy()
        pair1.destroy()
        pair2.destroy()
        return {"output_pair": new_pair, "swapping_station": self.station}


class DiscardQubitEvent(Event):
    """Event to discard a qubit and associated pair.

    For example if the qubit sat in memory too long and is discarded.

    Additional information in return dict of resolve method: None

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
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None

    Attributes
    ----------
    qubit

    """

    def __init__(
        self, time, qubit, priority=39, ignore_blocked=True, callback_functions=None
    ):
        self.qubit = qubit
        super(DiscardQubitEvent, self).__init__(
            time=time,
            required_objects=[self.qubit],
            priority=priority,
            ignore_blocked=ignore_blocked,
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(time={self.time}, qubit={self.qubit}, priority={self.priority}, ignore_blocked={self.ignore_blocked}), "
            + f"callback_functions={self._callback_functions}"
        )

    def __str__(self):
        return f"{self.__class__.__name__} at time={self.time} to discard {self.qubit.label}."

    def _main_effect(self):
        """Discards the qubit and associated pair, if the qubit still exists.

        Returns
        -------
        dict
            The return_dict of this event is updated with this.

        """
        qubit_pair = self.qubit.higher_order_object
        if qubit_pair is not None:
            for qubit in qubit_pair.qubits:
                qubit.destroy()
            qubit_pair.destroy()
        else:
            self.qubit.destroy()
        return {}


class EntanglementPurificationEvent(Event):
    """Perform a step of an entanglement purification protocol.

    Additional information in return dict of resolve method:

    "output_pair" : Pair
        The output pair of the entanglement purification step.
    "is_successful" : bool
        True if the entanglement purification was successful,
        false if not.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    pairs : list of Pairs
        The pairs involved in the entanglement purification protocol. Make
        sure these are at the correct stations and have the same qubit ordering.
    communication_time : scalar
        how long it takes for the result of the protocol to be communcated
        the remaining pair will be blocked for that amount of time
    protocol : {"dejmps"} or callable
        Can be one of the pre-defined or an arbitrary callable that takes
        a tensor product of pair states as input and returns a tuple of
        (success probability, state of a single pair) back.
        So far only supports n->1 protocols.
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None

    Attributes
    ----------
    pairs
    protocol
    communication_time

    """

    def __init__(
        self,
        time,
        pairs,
        communication_time,
        protocol="dejmps",
        callback_functions=None,
    ):
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
        self.communication_time = communication_time
        super(EntanglementPurificationEvent, self).__init__(
            time=time,
            required_objects=self.pairs
            + [qubit for pair in self.pairs for qubit in pair.qubits],
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(time={self.time}, pairs={self.pairs}, protocol={self.protocol}, "
            + f"communication_time={self.communication_time}), "
            + f"callback_functions={self._callback_functions}"
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
        dict
            The return_dict of this event is updated with this.

        """
        # probably could use a check that pairs are between same stations?
        for pair in self.pairs:
            pair.update_time()
        rho = mat.tensor(*[pair.state for pair in self.pairs])
        p_suc, state = self.protocol(rho)
        output_pair = self.pairs[0]
        output_pair.state = state
        output_pair.is_blocked = True
        output_pair.qubit1.is_blocked = True
        output_pair.qubit2.is_blocked = True
        for pair in self.pairs[1:]:  # pairs that have been destroyed in the process
            pair.qubits[0].destroy()
            pair.qubits[1].destroy()
            pair.destroy()
        if np.random.random() <= p_suc:  # if successful
            unblock_event = UnblockEvent(
                time=self.time + self.communication_time,
                quantum_objects=[output_pair, output_pair.qubit1, output_pair.qubit2],
            )
            self.event_queue.add_event(unblock_event)
            return {"output_pair": output_pair, "is_successful": True}
        else:  # if unsuccessful

            def destroy_function():
                output_pair.destroy()
                output_pair.qubits[0].destroy()
                output_pair.qubits[1].destroy()
                return {
                    "destroyed_objects": [
                        output_pair,
                        output_pair.qubits[0],
                        output_pair.qubits[1],
                    ]
                }

            destroy_event = GenericEvent(
                time=self.time + self.communication_time,
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

    Additional information in return dict of resolve method:

    "unblocked_objects" : list of WorldObject
        All objects that were unblocked by this event.


    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    quantum_objects : list of QuantumObjects
        The quantum objects to be unblocked.
    priority : int (expected 0...39)
        Default: 0 (because unblocking should happen as soon as possible)
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None

    Attributes
    ----------
    quantum_objects

    """

    def __init__(self, time, quantum_objects, priority=0, callback_functions=None):
        self.quantum_objects = quantum_objects
        super(UnblockEvent, self).__init__(
            time=time,
            required_objects=self.quantum_objects,
            priority=priority,
            ignore_blocked=True,
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(time={self.time}, quantum_objects={self.quantum_objects}, priority={self.priority}), "
            + f"callback_functions={self._callback_functions}"
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
