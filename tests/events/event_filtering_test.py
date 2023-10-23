from requsim.world import World
from requsim.events import Event, DiscardQubitEvent


class DummyEvent(Event):
    def __init__(
        self, time, required_objects=None, ignore_blocked=False, callback_functions=None
    ):
        if required_objects is None:
            required_objects = []
        if callback_functions is None:
            callback_functions = []
        super(DummyEvent, self).__init__(
            time,
            required_objects=required_objects,
            ignore_blocked=ignore_blocked,
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return ""

    def _main_effect(self):
        pass


def test_remove_by_condition_general():
    world = World()
    events = []
    for point_in_time in range(30):
        new_event = DummyEvent(time=point_in_time)
        world.event_queue.add_event(new_event)
        events += [new_event]
    # remove all events after time 15
    removed_events = world.event_queue.remove_by_condition(lambda x: x.time > 15)
    assert removed_events == events[16:]
    for event in events[:16]:
        assert event in world.event_queue.queue
    for event in events[16:]:
        assert event not in world.event_queue.queue
    # remove events at even times
    remaining_events = events[:16]
    removed_events = world.event_queue.remove_by_condition(lambda x: x.time % 2 == 0)
    assert removed_events == remaining_events[::2]
    for event in remaining_events[1::2]:
        assert event in world.event_queue.queue
    for event in remaining_events[::2]:
        assert event not in world.event_queue.queue


def test_remove_by_condition_example():
    pass


def test_recurring_filter():
    pass
