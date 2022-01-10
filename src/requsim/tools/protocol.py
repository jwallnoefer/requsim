import sys
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ..events import SourceEvent
from ..libs.aux_functions import distance


class Protocol(ABC):
    """Abstract base class for protocols.

    Parameters
    ----------
    world : World
        The world in which the protocol will be performed.

    Attributes
    ----------
    world

    """

    def __init__(self, world):
        self.world = world

    @abstractmethod
    def setup(self):
        """Setup function to be called after the world has been initialized.

        Should analyze the world to see if the protocol is applicable to the
        situation and possibly label stations/sources so they are easy
        to access in the check method of the protocol.
        """
        pass

    @abstractmethod
    def check(self):
        """The main method of the protocol.

        Should analyze the current status of the world and event_queue to
        make decisions about next steps.
        """
        pass


class MessageReadingProtocol(Protocol):
    """Abstract Protocol that can use additional information."""

    @abstractmethod
    def check(self, message=None):
        """Short summary.

        Parameters
        ----------
        message : None or dict
            Optional additional information for the Protocol to consider.
            Default is None.

        """
        pass


class TwoLinkProtocol(Protocol):
    """A class that collects various useful methods for two-link scenarios.

    But it is still abstract and misses the central check method.
    """

    def __init__(self, world, communication_speed):
        self.time_list = []
        self.state_list = []
        self.communication_speed = communication_speed
        super(TwoLinkProtocol, self).__init__(world=world)

    @property
    def data(self):
        return pd.DataFrame({"time": self.time_list, "state": self.state_list})

    def setup(self):
        """Identifies the stations and sources in the world.

        Should be run after the relevant WorldObjects have been added
        to the world.

        Returns
        -------
        None

        """
        stations = self.world.world_objects["Station"]
        assert len(stations) == 3
        if isinstance(stations[0].position, int):
            self.station_A, self.station_central, self.station_B = sorted(
                stations, key=lambda x: x.position
            )
        else:
            self.station_A, self.station_central, self.station_B = stations
        sources = self.world.world_objects["Source"]
        assert len(sources) == 2
        self.source_A = next(
            filter(
                lambda source: self.station_A in source.target_stations
                and self.station_central in source.target_stations,
                sources,
            )
        )
        self.source_B = next(
            filter(
                lambda source: self.station_central in source.target_stations
                and self.station_B in source.target_stations,
                sources,
            )
        )
        assert callable(
            getattr(self.source_A, "schedule_event", None)
        )  # schedule_event is a required method for this protocol
        assert callable(getattr(self.source_B, "schedule_event", None))

    def _get_left_pairs(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(
            filter(
                lambda x: x.is_between_stations(self.station_A, self.station_central),
                pairs,
            )
        )

    def _get_right_pairs(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(
            filter(
                lambda x: x.is_between_stations(self.station_central, self.station_B),
                pairs,
            )
        )

    def _get_long_range_pairs(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(
            filter(
                lambda x: x.is_between_stations(self.station_A, self.station_B),
                pairs,
            )
        )

    def _left_pairs_scheduled(self):
        return list(
            filter(
                lambda event: (
                    isinstance(event, SourceEvent)
                    and (self.station_A in event.source.target_stations)
                    and (self.station_central in event.source.target_stations)
                ),
                self.world.event_queue.queue,
            )
        )

    def _right_pairs_scheduled(self):
        return list(
            filter(
                lambda event: (
                    isinstance(event, SourceEvent)
                    and (self.station_central in event.source.target_stations)
                    and (self.station_B in event.source.target_stations)
                ),
                self.world.event_queue.queue,
            )
        )

    def _eval_pair(self, long_range_pair):
        comm_distance = np.max(
            [
                distance(self.station_central, self.station_A),
                distance(self.station_B, self.station_central),
            ]
        )
        comm_time = comm_distance / self.communication_speed

        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.state_list += [long_range_pair.state]
        return
