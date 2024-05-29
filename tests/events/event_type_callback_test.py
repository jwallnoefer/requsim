# events_test already tests that this works generally, but here some corner cases are checked
import numpy as np
import pytest

from requsim.events import Event, EventQueue
from unittest.mock import MagicMock


class DummyEvent(Event):
    def __init__(
        self, time, required_objects=[], ignore_blocked=False, callback_functions=[]
    ):
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


class OtherDummyEvent(Event):
    def __init__(
        self, time, required_objects=[], ignore_blocked=False, callback_functions=[]
    ):
        super(OtherDummyEvent, self).__init__(
            time,
            required_objects=required_objects,
            ignore_blocked=ignore_blocked,
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return ""

    def _main_effect(self):
        pass


class DummySubclassAEvent(DummyEvent):
    def __init__(
        self, time, required_objects=[], ignore_blocked=False, callback_functions=[]
    ):
        super(DummySubclassAEvent, self).__init__(
            time,
            required_objects=required_objects,
            ignore_blocked=ignore_blocked,
            callback_functions=callback_functions,
        )


class DummySubclassBEvent(DummyEvent):
    def __init__(
        self, time, required_objects=[], ignore_blocked=False, callback_functions=[]
    ):
        super(DummySubclassBEvent, self).__init__(
            time,
            required_objects=required_objects,
            ignore_blocked=ignore_blocked,
            callback_functions=callback_functions,
        )


@pytest.fixture
def event_queue():
    return EventQueue()


def test_only_specified_events_affected(event_queue):
    subclass_a_callback = MagicMock()
    event_queue.add_event_type_callback(DummySubclassAEvent, subclass_a_callback)
    event_queue.add_event(DummyEvent(time=np.random.random()))
    event_queue.add_event(DummyEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassAEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassAEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassAEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassBEvent(time=np.random.random()))
    event_queue.add_event(OtherDummyEvent(time=np.random.random()))
    subclass_a_added_after_callback = MagicMock()
    event_queue.add_event_type_callback(
        DummySubclassAEvent, subclass_a_added_after_callback
    )

    a_count = 0
    while event_queue.queue:
        if isinstance(event_queue.next_event, DummySubclassAEvent):
            a_count += 1
        event_queue.resolve_next_event()
        assert subclass_a_callback.call_count == a_count
        assert subclass_a_added_after_callback.call_count == a_count


def test_callback_affects_subclasses(event_queue):
    dummy_event_callback = MagicMock()
    event_queue.add_event_type_callback(DummyEvent, dummy_event_callback)
    event_queue.add_event(DummyEvent(time=np.random.random()))
    event_queue.add_event(DummyEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassAEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassAEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassAEvent(time=np.random.random()))
    event_queue.add_event(DummySubclassBEvent(time=np.random.random()))
    event_queue.add_event(OtherDummyEvent(time=np.random.random()))
    event_queue.add_event(OtherDummyEvent(time=np.random.random()))
    dummy_event_added_after_callback = MagicMock()
    event_queue.add_event_type_callback(DummyEvent, dummy_event_added_after_callback)

    count = 0
    while event_queue.queue:
        if not isinstance(event_queue.next_event, OtherDummyEvent):
            count += 1
        event_queue.resolve_next_event()
        assert dummy_event_callback.call_count == count
        assert dummy_event_added_after_callback.call_count == count
