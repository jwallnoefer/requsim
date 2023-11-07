from requsim.quantum_objects import Station, Pair, Qubit
from requsim.world import World
from requsim.events import Event, DiscardQubitEvent, EntanglementSwappingEvent
import requsim.libs.matrix as mat


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
    # the most frequent use case for this has been removing DiscardQubitEvent that are no longer needed
    # because for some parameter sets these really pile up excessively
    world = World()
    stations = [
        Station(world=world, position=i * 100, memory_cutoff_time=20) for i in range(4)
    ]
    pairs = []
    rho_phiplus = mat.phiplus @ mat.H(mat.phiplus)
    for left, right in zip(stations[:-1], stations[1:]):
        pair = Pair(
            world=world,
            qubits=[left.create_qubit(), right.create_qubit()],
            initial_state=rho_phiplus,
        )
        pairs.append(pair)
    assert (
        len(
            list(
                filter(lambda x: x.type == "DiscardQubitEvent", world.event_queue.queue)
            )
        )
        == 6
    )
    new_event = EntanglementSwappingEvent(time=1, pairs=pairs[0:2], station=stations[1])
    world.event_queue.add_event(new_event)
    world.event_queue.resolve_next_event()
    assert pairs[0].qubits[1] not in world
    assert pairs[1].qubits[0] not in world
    obsolete_event_1 = pairs[0].qubits[1].required_by_events[0]
    assert obsolete_event_1.type == "DiscardQubitEvent"
    assert obsolete_event_1 in world.event_queue.queue
    obsolete_event_2 = pairs[1].qubits[0].required_by_events[0]
    assert obsolete_event_2.type == "DiscardQubitEvent"
    assert obsolete_event_2 in world.event_queue.queue
    world.event_queue.remove_by_condition(
        lambda x: x.type == "DiscardQubitEvent" and not x.req_objects_exist()
    )
    assert obsolete_event_1 not in world.event_queue.queue
    assert obsolete_event_2 not in world.event_queue.queue
    # but other events are untouched
    assert (
        len(
            list(
                filter(lambda x: x.type == "DiscardQubitEvent", world.event_queue.queue)
            )
        )
        == 4
    )


def test_recurring_filter():
    world = World()
    qubits = []
    discard_events = []
    for point_in_time in range(100):
        new_event = DummyEvent(time=point_in_time)
        world.event_queue.add_event(new_event)
        if point_in_time % 10 == 0:
            qubit = Qubit(world=world)
            qubits += [qubit]
            new_event = DiscardQubitEvent(time=point_in_time + 0.5, qubit=qubit)
            world.event_queue.add_event(new_event)
            discard_events += [new_event]
    interval = 5
    world.event_queue.add_recurring_filter(
        condition=lambda x: x.type == "DiscardQubitEvent" and not x.req_objects_exist(),
        filter_interval=interval,
    )
    for i in range(interval + 1):
        world.event_queue.resolve_next_event()
    # destroy some qubits
    associated_events = []
    for qubit in qubits[-2:]:
        assert qubit in world
        associated_event = qubit.required_by_events[0]
        assert associated_event in world.event_queue.queue
        associated_events += [associated_event]
        qubit.destroy()
    for i in range(interval):
        world.event_queue.resolve_next_event()
    for associated_event in associated_events:
        assert associated_event in world.event_queue.queue
    world.event_queue.resolve_next_event()
    for associated_event in associated_events:
        assert associated_event not in world.event_queue.queue
