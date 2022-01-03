from .events import EventQueue
from collections import defaultdict


class World(object):
    """A collection of WorldObjects with an EventQueue.

    The World can be understood as a central object describing an experimental
    setup. It keeps track of all WorldObjects, which contain all the information
    a protocol can access to make decisions.

    Attributes
    ----------
    event_queue : EventQueue
        A schedule of events so they can be resolved in order.
    world_objects : dict
        keys: str of world object type
        values: list of WorldObjects with this type

    """

    def __init__(self):
        self.event_queue = EventQueue()
        # world_objects collects everything about the current state of the world
        self.world_objects = {}
        self._label_counters = defaultdict(lambda: 0)

    def __contains__(self, world_object):
        return world_object in self.world_objects[world_object.type]

    def print_status(self, filter=None, max_display=12):
        """Print scheduled events and objects in a somewhat formatted way.

        Parameters
        ----------
        filter : {"all", "Event", str, None}
            None displays `max_display` entries for events and each object type.
            "all" displays all events and objects regardless of `max_display`.
            "Event" displays all events.
            If any other str, will display all objects with matching object
            type. e.g. "Pair" or "Station"
        max_display : int
            If filter is None, only `max_display` items per category will be
            displayed.

        Returns
        -------
        None

        """
        if filter is None:
            print("Event queue:")
            num_events = len(self.event_queue.queue)
            if num_events > max_display:
                print(
                    f'There are {num_events} in the event queue. Use world.print_status("Event") to display all. Here are the first {max_display}:'
                )
            for event in self.event_queue.queue[:max_display]:
                print(event)
            print("%================%")
            print("Objects")
            for k, v in self.world_objects.items():
                print("------")
                print(k + ":")
                if len(v) > max_display:
                    print(
                        f'There are {len(v)} {k}-Objects. Use world.print_status("{k}") to display all. Here are the first {max_display}:'
                    )
                for obj in v[:max_display]:
                    print(obj)
        elif filter == "all":
            print("Event queue:")
            for event in self.event_queue.queue:
                print(event)
            print("%================%")
            print("Objects")
            for k, v in self.world_objects.items():
                print("------")
                print(k + ":")
                for obj in v:
                    print(obj)
        elif filter == "Event":
            print("Event queue:")
            for event in self.event_queue.queue:
                print(event)
        else:
            if filter not in self.world_objects:
                print(f"There are no objects of type {filter}.")
            else:
                print(filter + ":")
                for obj in self.world_objects[filter]:
                    print(obj)

    def register_world_object(self, world_object):
        """Add a WorldObject to this world.

        Parameters
        ----------
        world_object : WorldObject
            WorldObject that should be added and tracked by `self.world_objects`

        Returns
        -------
        str
            A label for the `world_object`, which is the object type plus a
            counter that ticks up every time an object of that type is created.

        """
        object_type = world_object.type
        if object_type not in self.world_objects:
            self.world_objects[object_type] = []
        self.world_objects[object_type] += [world_object]
        self._label_counters[object_type] += 1
        return f"{object_type} {self._label_counters[object_type]}"

    def deregister_world_object(self, world_object):
        """Remove a WorldObject from this World.

        Parameters
        ----------
        world_object : WorldObject
            The WorldObject that is removed from this World.

        Returns
        -------
        None

        """
        object_type = world_object.type
        type_list = self.world_objects[object_type]
        try:
            type_list.remove(world_object)
        except ValueError:  # happens if object has been removed already
            pass
