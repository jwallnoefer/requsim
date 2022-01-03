import sys
from abc import ABC


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
        self._destroy_callbacks = []
        self.add_destroy_callback(self.world.deregister_world_object)

    def __str__(self):
        return self.label

    def add_destroy_callback(self, callback_func):
        self._destroy_callbacks += [callback_func]

    def destroy(self):
        """Remove this WorldObject from the world."""
        # in the future associated events may also be removed by subscribing to destruction
        for callback_func in self._destroy_callbacks:
            callback_func(self)

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
