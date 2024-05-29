from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from warnings import warn

import numpy as np
import pandas as pd

from requsim.events import EntanglementSwappingEvent, SourceEvent
from requsim.libs.aux_functions import distance
from requsim.tools.protocol import Protocol


C = 2e8  # speed of light in optical fiber
L_ATT = 22000  # Attenuation length in optical fiber


@lru_cache(maxsize=int(5e4))
def is_event_swapping_pairs(event, pair1, pair2):
    return (
        isinstance(event, EntanglementSwappingEvent)
        and (pair1 in event.pairs)
        and (pair2 in event.pairs)
    )


@lru_cache(maxsize=int(5e4))
def is_sourceevent_between_stations(event, station1, station2):
    return (
        isinstance(event, SourceEvent)
        and (station1 in event.source.target_stations)
        and (station2 in event.source.target_stations)
    )


class BaseManylinkProtocol(Protocol):
    def __init__(self):
        self.stations = []
        self.sources = []
        self.num_memories = None
        self.communication_speed = None
        # derived quantities for easy access
        self.link_stations = []
        self.host_station_by_source = {}
        self.sources_by_station = defaultdict(list)
        # protocol internal
        self.time_list = []
        self.state_list = []
        self.scheduled_swappings = defaultdict(list)
        super(BaseManylinkProtocol, self).__init__()

    def setup(self, world, stations, sources, num_memories=1, communication_speed=C):
        self.world = world
        self.stations = stations
        self.sources = sources
        self.num_memories = num_memories
        self.communication_speed = communication_speed
        # Station ordering left to right
        num_stations = len(self.stations)
        num_sources = len(self.sources)
        assert num_sources == num_stations - 1
        for source in self.sources:
            assert callable(
                getattr(source, "schedule_event", None)
            )  # schedule_event is a required method for this protocol
        self.link_stations = [
            [self.stations[i], self.stations[i + 1]] for i in range(num_sources)
        ]
        for i, source in enumerate(self.sources):
            self.host_station_by_source[source] = self.stations[2 * (i // 2) + 1]
        for source in self.sources:
            self.sources_by_station[source.target_stations[0]] += [source]
            self.sources_by_station[source.target_stations[1]] += [source]

    @property
    def data(self):
        return pd.DataFrame({"time": self.time_list, "state": self.state_list})

    def _get_pairs_between_stations(self, station1, station2):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(
            filter(lambda pair: pair.is_between_stations(station1, station2), pairs)
        )

    def _get_pairs_scheduled(self, station1, station2):
        return list(
            filter(
                lambda event: is_sourceevent_between_stations(
                    event, station1, station2
                ),
                self.world.event_queue.queue,
            )
        )

    def _eval_pair(self, long_range_pair):
        comm_distance = np.max(
            [
                distance(self.stations[0], self.stations[-2]),
                distance(self.stations[-1], self.stations[1]),
            ]
        )
        # comm_distance is simple upper limit for swapping communication
        comm_time = comm_distance / self.communication_speed

        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.state_list += [long_range_pair.state]
        return

    def pairs_at_station(self, station):
        station_index = self.stations.index(station)
        pairs_left = []
        pairs_right = []
        for qubit in station.qubits:
            pair = qubit.higher_order_object
            qubit_list = list(pair.qubits)
            qubit_list.remove(qubit)
            qubit_neighbor = qubit_list[0]
            if self.stations.index(qubit_neighbor._info["station"]) < station_index:
                pairs_left += [pair]
            else:
                pairs_right += [pair]
        return (pairs_left, pairs_right)

    def memory_check(self, station):
        station_index = self.stations.index(station)
        free_memories_left = self.num_memories
        free_memories_right = self.num_memories
        pairs_left, pairs_right = self.pairs_at_station(station)
        free_memories_left -= len(pairs_left)
        free_memories_right -= len(pairs_right)
        free_memories_left -= len(
            self._get_pairs_scheduled(self.stations[station_index - 1], station)
        )
        free_memories_right -= len(
            self._get_pairs_scheduled(station, self.stations[station_index + 1])
        )
        return (free_memories_left, free_memories_right)

    def _check_station_overflow(self, station):
        left_pairs, right_pairs = self.pairs_at_station(station)
        has_overflowed = False
        if len(left_pairs) > self.num_memories:
            last_pair = left_pairs[-1]
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy()
            has_overflowed = True
        if len(right_pairs) > self.num_memories:
            last_pair = right_pairs[-1]
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy()
            has_overflowed = True
        return has_overflowed

    def _check_new_source_events(self, station):
        sources_to_check = self.sources_by_station[station]
        for source in sources_to_check:
            host_station = self.host_station_by_source[source]
            free_left, free_right = self.memory_check(host_station)
            for _ in range(free_left):
                self.sources_by_station[host_station][0].schedule_event()
            for _ in range(free_right):
                self.sources_by_station[host_station][1].schedule_event()

    def _check_swapping(self, station):
        left_pairs, right_pairs = self.pairs_at_station(station)
        num_swappings = min(len(left_pairs), len(right_pairs))
        if num_swappings:
            # get rid of events that are no longer scheduled
            self.scheduled_swappings[station] = [
                event
                for event in self.scheduled_swappings[station]
                if event in self.world.event_queue.queue
            ]
        for left_pair, right_pair in zip(
            left_pairs[:num_swappings], right_pairs[:num_swappings]
        ):
            # assert that we do not schedule the same swapping more than once
            try:
                next(
                    filter(
                        lambda event: is_event_swapping_pairs(
                            event, left_pair, right_pair
                        ),
                        self.scheduled_swappings[station],
                    )
                )
                is_already_scheduled = True
            except StopIteration:
                is_already_scheduled = False
            if not is_already_scheduled:
                ent_swap_event = EntanglementSwappingEvent(
                    time=self.world.event_queue.current_time,
                    pairs=[left_pair, right_pair],
                    station=station,
                )
                self.scheduled_swappings[station] += [ent_swap_event]
                self.world.event_queue.add_event(ent_swap_event)

    def _check_long_distance_pair(self):
        # Evaluate long range pairs
        long_range_pairs = self._get_pairs_between_stations(
            self.stations[0], self.stations[-1]
        )
        if long_range_pairs:
            for long_range_pair in long_range_pairs:
                self._eval_pair(long_range_pair)
                # cleanup
                long_range_pair.qubits[0].destroy()
                long_range_pair.qubits[1].destroy()
                long_range_pair.destroy()
            # self.check()  # was useful at some point for other scenarios

    @abstractmethod
    def check(self, message=None):
        raise NotImplementedError


class DefaultManylinkProtocol(BaseManylinkProtocol):
    def check(self, message=None):
        if message is None:
            for station in self.stations:
                self._check_station_overflow(station)
            for station in self.stations:
                self._check_new_source_events(station)
            for station in self.stations:
                self._check_swapping(station)
            self._check_long_distance_pair()
        elif (
            message["event_type"] == "SourceEvent"
            and message["resolve_successful"] is True
        ):
            output_pair = message["output_pair"]
            stations = [
                output_pair.qubit1._info["station"],
                output_pair.qubit2._info["station"],
            ]
            for station in stations:
                has_overflowed = self._check_station_overflow(station)
                if has_overflowed:
                    self._check_new_source_events(station)
                self._check_swapping(station)
        elif (
            message["event_type"] == "SourceEvent"
            and message["resolve_successful"] is False
        ):
            warn("A SourceEvent has resolved unsuccessfully. This should never happen.")
        elif (
            message["event_type"] == "DiscardQubitEvent"
            and message["resolve_successful"] is True
        ):
            discarded_qubit = message["qubit"]
            self._check_new_source_events(discarded_qubit._info["station"])
        elif (
            message["event_type"] == "DiscardQubitEvent"
            and message["resolve_successful"] is False
        ):
            pass
        elif (
            message["event_type"] == "EntanglementSwappingEvent"
            and message["resolve_successful"] is True
        ):
            self._check_new_source_events(message["swapping_station"])
            output_pair = message["output_pair"]
            for station in [
                output_pair.qubit1._info["station"],
                output_pair.qubit2._info["station"],
            ]:
                self._check_swapping(station)
            self._check_long_distance_pair()
        elif (
            message["event_type"] == "EntanglementSwappingEvent"
            and message["resolve_successful"] is False
        ):
            # warn("An EntanglementSwappingEvent has resolved unsuccessfully. Trying to recover.")
            # for station in self.stations:
            #     self._check_swapping(station)
            pass
        else:
            warn(f"Unrecognized message type encountered: {message}")


class CustomManylinkProtocol(BaseManylinkProtocol):
    def __init__(self):
        self.step = 0
        self.counter = 0
        super(CustomManylinkProtocol, self).__init__()

    def check(self, message=None):
        if len(self.world.event_queue.queue) != 0:
            return
        if self.step == 0:
            for source in self.sources:
                source.schedule_event()
            self.step = 1
        elif self.step == 1:
            pairs = self.world.world_objects["Pair"]
            if len(pairs) == 1:
                self._check_long_distance_pair()
                self.step = 0
                self.counter = 0
                self.check()
            else:
                self._check_swapping(self.stations[1 + self.counter])
                self.counter += 1


class ObserveOnlyManylinkProtocol(BaseManylinkProtocol):
    # This one does not use any information about what just happened.
    def check(self, message=None):
        for station in self.stations:
            self._check_station_overflow(station)
        for station in self.stations:
            self._check_new_source_events(station)
        for station in self.stations:
            self._check_swapping(station)
        self._check_long_distance_pair()


class CompositeProtocol(BaseManylinkProtocol):
    def __init__(self, subprotocol):
        self.subprotocol = subprotocol
        self.subprotocols = []
        super(CompositeProtocol, self).__init__()

    def setup(
        self,
        world,
        stations,
        sources,
        num_memories=1,
        communication_speed=C,
    ):
        super(CompositeProtocol, self).setup(
            world=world,
            stations=stations,
            sources=sources,
            num_memories=num_memories,
            communication_speed=communication_speed,
        )
        if np.isscalar(self.num_memories):
            num_memories = [self.num_memories] * len(self.stations)
        else:
            assert len(self.num_memories) == len(self.stations)
            num_memories = self.num_memories
        self.subprotocols = [deepcopy(self.subprotocol) for i in range(len(stations))]
        for i, (station, n_memories, subprotocol) in enumerate(
            zip(self.stations, num_memories, self.subprotocols)
        ):
            if i % 2 == 0:
                sources = []
            else:
                sources = [self.sources[i - 1], self.sources[i]]
            subprotocol.setup(
                world=world,
                station=station,
                sources_at_station=sources,
                num_memories=n_memories,
            )

    def check(self, message=None):
        if message is None:
            for protocol in self.subprotocols:
                protocol.check(message=None)
        elif (
            message["event_type"] == "EntanglementSwappingEvent"
            and message["resolve_successful"] is True
        ):
            output_pair = message["output_pair"]
            if output_pair.is_between_stations(self.stations[0], self.stations[-1]):
                self._eval_pair(output_pair)
                output_pair.qubits[0].destroy()
                output_pair.qubits[1].destroy()
                output_pair.destroy()
        elif (
            message["event_type"] == "DiscardQubitEvent"
            and message["resolve_successful"] is True
        ):
            for protocol in self.subprotocols:
                protocol.check(message=message)


class LocalProtocol(object):
    def __init__(self):
        # attributes describing the scenario and options once set up
        self.station = None
        self.num_memories = None
        self.sources_at_station = []
        # derived quantities
        self.event_tracking = defaultdict(list)
        self.associated_sources = []
        # protocol internal
        self.memory_tracking = {"left": [], "right": []}
        self._swapping_data = []

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def setup(self, world, station, num_memories=1, sources_at_station=None):
        self.world = world
        self.station = station
        self.station.protocol = self
        self.num_memories = num_memories
        if sources_at_station is None:
            self.sources_at_station = []
        else:
            self.sources_at_station = sources_at_station
        self.associated_sources = list(
            filter(
                lambda s: station in s.target_stations,
                station.world.world_objects["Source"],
            )
        )

    def _track_busy_mem_after_swapping(self, event_return_dict):
        if (
            event_return_dict["event_type"] == "EntanglementSwappingEvent"
            and event_return_dict["resolve_successful"]
        ):
            num_busy_mem = len(self.memory_tracking["left"]) + len(
                self.memory_tracking["right"]
            )
            self._swapping_data += [num_busy_mem]

    def _untrack_event(self, event_return_dict):
        event = event_return_dict["event"]
        self.event_tracking[event.type].remove(event)

    def _untrack_qubit_from_memory(self, qubit):
        if qubit in self.memory_tracking["left"]:
            self.memory_tracking["left"].remove(qubit)
        if qubit in self.memory_tracking["right"]:
            self.memory_tracking["right"].remove(qubit)

    def assign_to_memory_callback(self, event_return_dict):
        assert event_return_dict["event_type"] == "SourceEvent"
        assert event_return_dict["resolve_successful"]
        event_source = event_return_dict["source"]
        source_index = self.associated_sources.index(event_source)
        new_pair = event_return_dict["output_pair"]
        for qubit in new_pair.qubits:
            if qubit in self.station.qubits:
                qubit.add_destroy_callback(self._untrack_qubit_from_memory)
                if source_index == 0:
                    self.memory_tracking["left"].append(qubit)
                elif source_index == 1:
                    self.memory_tracking["right"].append(qubit)

    @property
    def left_pairs(self):
        return [qubit.higher_order_object for qubit in self.memory_tracking["left"]]

    @property
    def right_pairs(self):
        return [qubit.higher_order_object for qubit in self.memory_tracking["right"]]

    def _check_station_overflow(self):
        left_qubits, right_qubits = (
            self.memory_tracking["left"],
            self.memory_tracking["right"],
        )

        has_overflowed = False
        if len(left_qubits) > self.num_memories:
            last_pair = left_qubits[-1].higher_order_object
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy()
            has_overflowed = True
        if len(right_qubits) > self.num_memories:
            last_pair = right_qubits[-1].higher_order_object
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy()
            has_overflowed = True
        return has_overflowed

    def recheck_swapping_on_failed(self, event_return_dict=None):
        if (
            event_return_dict["event_type"] == "EntanglementSwappingEvent"
            and not event_return_dict["resolve_successful"]
        ):
            self.check_swapping(event_return_dict)

    def check_scheduling(self, event_return_dict=None):
        for source in self.sources_at_station:
            source_index = self.associated_sources.index(source)
            if source_index == 0:
                active_memories = len(self.memory_tracking["left"])
            elif source_index == 1:
                active_memories = len(self.memory_tracking["right"])
            scheduled_busy = len(
                list(
                    filter(
                        lambda x: x.source == source, self.event_tracking["SourceEvent"]
                    )
                )
            )
            busy_memories = active_memories + scheduled_busy
            for i in range(self.num_memories - busy_memories):
                event = source.schedule_event()
                self.event_tracking[event.type] += [event]
                event.add_callback(self._untrack_event)
                for station in source.target_stations:
                    event.add_callback(station.protocol.assign_to_memory_callback)
                for station in source.target_stations:
                    event.add_callback(station.protocol.check_swapping)

    def check_swapping(self, event_return_dict=None):
        # limited by the shorter list - zip does that automatically
        has_overflowed = self._check_station_overflow()
        if has_overflowed:
            for source in self.associated_sources:
                for station in source.target_stations:
                    station.protocol.check_scheduling()
        for left_qubit, right_qubit in zip(
            self.memory_tracking["left"], self.memory_tracking["right"]
        ):
            left_pair = left_qubit.higher_order_object
            right_pair = right_qubit.higher_order_object
            # avoid scheduling twice
            try:
                next(
                    filter(
                        lambda event: is_event_swapping_pairs(
                            event, left_pair, right_pair
                        ),
                        self.event_tracking["EntanglementSwappingEvent"],
                    )
                )
                is_already_scheduled = True
            except StopIteration:
                is_already_scheduled = False
            if not is_already_scheduled:
                ent_swap_event = EntanglementSwappingEvent(
                    time=self.world.event_queue.current_time,
                    pairs=[left_pair, right_pair],
                    station=self.station,
                )
                ent_swap_event.add_callback(self.check_scheduling)
                # swap events fail often when multiple swappings were scheduled simultaneously
                ent_swap_event.add_callback(self.recheck_swapping_on_failed)
                self.event_tracking[ent_swap_event.type] += [ent_swap_event]
                ent_swap_event.add_callback(self._untrack_event)
                self.world.event_queue.add_event(ent_swap_event)

    def check(self, message=None):
        self.check_scheduling(event_return_dict=message)


class CallbackProtocol(BaseManylinkProtocol):
    def _check_new_source_events(self, station):
        sources_to_check = self.sources_by_station[station]
        for source in sources_to_check:
            host_station = self.host_station_by_source[source]
            free_left, free_right = self.memory_check(host_station)
            for _ in range(free_left):
                new_event = self.sources_by_station[host_station][0].schedule_event()
                new_event.add_callback(self.on_source_event_callback)
            for _ in range(free_right):
                new_event = self.sources_by_station[host_station][1].schedule_event()
                new_event.add_callback(self.on_source_event_callback)

    def on_source_event_callback(self, event_return_dict=None):
        if event_return_dict is not None:
            stations_to_check = event_return_dict["source"].target_stations
            for station in stations_to_check:
                has_overflowed = self._check_station_overflow(station)
                if has_overflowed:
                    self._check_new_source_events(station)
            for station in stations_to_check:
                self._check_swapping(station)

    def on_swapping_event_callback(self, event_return_dict=None):
        if event_return_dict["resolve_successful"]:
            output_pair = event_return_dict["output_pair"]
            if output_pair.is_between_stations(self.stations[0], self.stations[-1]):
                self._eval_pair(output_pair)
                # cleanup
                output_pair.qubits[0].destroy()
                output_pair.qubits[1].destroy()
                output_pair.destroy()
            station = event_return_dict["swapping_station"]
            self._check_new_source_events(station)
        else:  # recheck on failed swapping
            station = event_return_dict["event"].station
            self._check_swapping(station)

    def _check_swapping(self, station):
        left_pairs, right_pairs = self.pairs_at_station(station)
        num_swappings = min(len(left_pairs), len(right_pairs))
        if num_swappings:
            # get rid of events that are no longer scheduled
            self.scheduled_swappings[station] = [
                event
                for event in self.scheduled_swappings[station]
                if event in self.world.event_queue.queue
            ]
        for left_pair, right_pair in zip(
            left_pairs[:num_swappings], right_pairs[:num_swappings]
        ):
            # assert that we do not schedule the same swapping more than once
            try:
                next(
                    filter(
                        lambda event: is_event_swapping_pairs(
                            event, left_pair, right_pair
                        ),
                        self.scheduled_swappings[station],
                    )
                )
                is_already_scheduled = True
            except StopIteration:
                is_already_scheduled = False
            if not is_already_scheduled:
                ent_swap_event = EntanglementSwappingEvent(
                    time=self.world.event_queue.current_time,
                    pairs=[left_pair, right_pair],
                    station=station,
                )
                self.scheduled_swappings[station] += [ent_swap_event]
                ent_swap_event.add_callback(self.on_swapping_event_callback)
                self.world.event_queue.add_event(ent_swap_event)

    def check(self, message=None):
        if message is None:  # expected on first call
            for station in self.stations:
                self._check_new_source_events(station)
