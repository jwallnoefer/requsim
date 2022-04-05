import unittest
from unittest.mock import MagicMock
from requsim.world import World
from requsim.events import (
    Event,
    SourceEvent,
    EntanglementSwappingEvent,
    EventQueue,
    DiscardQubitEvent,
    EntanglementPurificationEvent,
    UnblockEvent,
)
from requsim.quantum_objects import Qubit, Pair, Station, Source, WorldObject
import numpy as np
import requsim.libs.matrix as mat
from requsim.libs.aux_functions import distance

_COMMUNICATION_SPEED = 2e8


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


class PriorityEvent(Event):
    def __init__(self, time, required_objects=[], priority=20):
        super(PriorityEvent, self).__init__(
            time=time, required_objects=required_objects, priority=priority
        )

    def __repr__(self):
        return ""

    def _main_effect(self):
        pass


class DummyObject(WorldObject):
    pass


def _random_test_state(n):
    test_state = np.random.random((2 ** n, 2 ** n))
    test_state = test_state + test_state.T  # symmetrize
    # normalize, so we have random real density matrix
    test_state = test_state / np.trace(test_state)
    return test_state


def _known_dejmps_identical_copies(lambdas):
    # known formula for two identical states diagonal in the bell basis
    lambda_00, lambda_01, lambda_10, lambda_11 = lambdas
    new_lambdas = [
        lambda_00**2 + lambda_11**2,
        lambda_01**2 + lambda_10**2,
        2 * lambda_00 * lambda_11,
        2 * lambda_01 * lambda_10,
    ]
    p_suc = np.sum(new_lambdas)
    new_lambdas = np.array(new_lambdas) / p_suc
    return p_suc, new_lambdas


def _bogus_epp(rho):
    p_suc = 1
    state_after = np.dot(mat.phiplus, mat.H(mat.phiplus))
    return p_suc, state_after


class TestEvents(unittest.TestCase):
    # Not sure what else we could test here that does not boil down to asking
    # is the code exactly the code?
    def setUp(self):
        self.world = World()

    def _aux_general_test(self, event):
        self.assertIsInstance(event, Event)

    def _check_event_return_is_valid(self, return_value):
        if return_value is None:
            return True
        elif isinstance(return_value, dict):
            self.assertIn("resolve_successful", return_value)
            self.assertIsInstance(return_value["resolve_successful"], bool)
            self.assertIn("event_type", return_value)
            self.assertIsInstance(return_value["event_type"], str)
            return True
        else:
            return False

    def _aux_test_event_with_callback(self, EventClass, should_require=[], **kwargs):
        kwargs = dict(kwargs)
        mock_init_callback = MagicMock()
        if "callback_functions" in kwargs:
            kwargs["callback_functions"] += [mock_init_callback]
        else:
            kwargs["callback_functions"] = [mock_init_callback]
        event = EventClass(**kwargs)
        self._aux_general_test(event)
        mock_added_callback = MagicMock()
        event.add_callback(mock_added_callback)
        # explicitly required objects should be required
        if "required_objects" in kwargs:
            for obj in kwargs["required_objects"]:
                self.assertIn(obj, event.required_objects)
                self.assertIn(event, obj.required_by_events)
        # check if event-specific objects that should be (maybe implicitly)
        # required, are indeed required
        for obj in should_require:
            self.assertIn(obj, event.required_objects)
        # implicitly required objects should know of the event
        for obj in event.required_objects:
            self.assertIn(event, obj.required_by_events)
        self.world.event_queue.add_event(event)
        mock_init_callback.assert_not_called()
        mock_added_callback.assert_not_called()
        return_value = self.world.event_queue.resolve_next_event()
        if "required_objects" in kwargs:
            for obj in kwargs["required_objects"]:
                self.assertNotIn(event, obj.required_by_events)
        # sometimes events do only implicitly specify the required objects
        for obj in event.required_objects:
            self.assertNotIn(event, obj.required_by_events)
        mock_init_callback.assert_called_once()
        mock_added_callback.assert_called_once()
        self.assertTrue(self._check_event_return_is_valid(return_value))

    def test_dummy_event(self):  # basically just tests the tracking of required objects
        test_objects = [DummyObject(world=self.world) for i in range(20)]
        self._aux_test_event_with_callback(
            DummyEvent, time=0, required_objects=test_objects
        )

    def test_source_event(self):
        test_station = Station(world=self.world, position=0)
        test_source = Source(
            world=self.world, position=0, target_stations=[test_station, test_station]
        )
        should_require = [test_source, test_station]
        self._aux_test_event_with_callback(
            SourceEvent,
            should_require=should_require,
            time=0,
            source=test_source,
            initial_state=MagicMock(),
        )
        # is there a pair after the source event?
        self.assertEqual(len(self.world.world_objects["Pair"]), 1)

    def test_entanglement_swapping_event(self):
        left_station = Station(world=self.world, position=0)
        middle_station = Station(world=self.world, position=100)
        right_station = Station(world=self.world, position=200)
        pair1 = Pair(
            world=self.world,
            qubits=[left_station.create_qubit(), middle_station.create_qubit()],
            initial_state=np.dot(mat.phiplus, mat.H(mat.phiplus)),
        )
        pair2 = Pair(
            world=self.world,
            qubits=[middle_station.create_qubit(), right_station.create_qubit()],
            initial_state=np.dot(mat.phiplus, mat.H(mat.phiplus)),
        )
        should_require = [pair1, pair2] + [
            qubit for pair in [pair1, pair2] for qubit in pair.qubits
        ]
        self._aux_test_event_with_callback(
            EntanglementSwappingEvent,
            should_require=should_require,
            time=0,
            pairs=[pair1, pair2],
            station=middle_station,
        )
        # functionality test is missing

    def test_discard_qubit_event(self):
        world = self.world
        # general test
        qubit = Qubit(world=world)
        self._aux_test_event_with_callback(
            DiscardQubitEvent, should_require=[qubit], time=0, qubit=qubit
        )
        # now test whether qubit actually gets discarded
        qubit = Qubit(world=world)
        event = DiscardQubitEvent(time=0, qubit=qubit)
        self.assertIn(qubit, world.world_objects[qubit.type])
        self.assertIn(qubit, world)
        event.resolve()
        self.assertNotIn(qubit, world.world_objects[qubit.type])
        self.assertNotIn(qubit, world)
        # now test whether the whole pair gets discarded if a qubit is discarded
        qubits = [Qubit(world=world) for i in range(2)]
        pair = Pair(world=world, qubits=qubits, initial_state=MagicMock())
        event = DiscardQubitEvent(time=0, qubit=qubits[0])
        self.assertIn(qubits[0], world.world_objects[qubits[0].type])
        self.assertIn(qubits[1], world.world_objects[qubits[1].type])
        self.assertIn(pair, world.world_objects[pair.type])
        event.resolve()
        self.assertNotIn(qubits[0], world.world_objects[qubits[0].type])
        self.assertNotIn(qubits[1], world.world_objects[qubits[1].type])
        self.assertNotIn(pair, world.world_objects[pair.type])

    def test_unblock_event(self):
        # general test
        test_object = WorldObject(world=self.world)
        test_object.is_blocked = True
        self._aux_test_event_with_callback(
            UnblockEvent,
            should_require=[test_object],
            time=0,
            quantum_objects=[test_object],
        )
        # test for functionality
        test_object = WorldObject(world=self.world)
        test_object.is_blocked = True
        event = UnblockEvent(time=0, quantum_objects=[test_object])
        self.world.event_queue.add_event(event)
        self.assertIs(test_object.is_blocked, True)
        self.world.event_queue.resolve_next_event()
        self.assertIs(test_object.is_blocked, False)

    def test_epp_general(self):  # more involved functionality test below
        station1 = Station(world=self.world, position=0)
        station2 = Station(world=self.world, position=100)
        pair1 = Pair(
            world=self.world,
            qubits=[station1.create_qubit(), station2.create_qubit()],
            initial_state=_random_test_state(n=2),
        )
        pair2 = Pair(
            world=self.world,
            qubits=[station1.create_qubit(), station2.create_qubit()],
            initial_state=_random_test_state(n=2),
        )
        should_require = [pair1, pair2] + pair1.qubits + pair2.qubits
        self._aux_test_event_with_callback(
            EntanglementPurificationEvent,
            should_require=should_require,
            time=0,
            pairs=[pair1, pair2],
            communication_time=5,
        )


class TestEPP(unittest.TestCase):
    # entanglement purification gets own test case because there is more to test
    def setUp(self):
        self.world = World()
        self.station1 = Station(world=self.world, position=0)
        self.station2 = Station(world=self.world, position=100)

    # #  first we test the built_in epp modes
    def test_dejmps_epp_event(self):
        # test with the nice blackbox formula for two identical pairs
        for i in range(4):
            lambdas = np.random.random(4)
            lambdas[0] = lambdas[0] + 3
            lambdas = lambdas / np.sum(lambdas)
            initial_state = (
                lambdas[0] * np.dot(mat.phiplus, mat.H(mat.phiplus))
                + lambdas[1] * np.dot(mat.psiplus, mat.H(mat.psiplus))
                + lambdas[2] * np.dot(mat.phiminus, mat.H(mat.phiminus))
                + lambdas[3] * np.dot(mat.psiminus, mat.H(mat.psiminus))
            )
            trusted_p_suc, trusted_lambdas = _known_dejmps_identical_copies(lambdas)
            trusted_state = (
                trusted_lambdas[0] * np.dot(mat.phiplus, mat.H(mat.phiplus))
                + trusted_lambdas[1] * np.dot(mat.psiplus, mat.H(mat.psiplus))
                + trusted_lambdas[2] * np.dot(mat.phiminus, mat.H(mat.phiminus))
                + trusted_lambdas[3] * np.dot(mat.psiminus, mat.H(mat.psiminus))
            )
            self.world.world_objects["Pair"] = []
            while not self.world.world_objects[
                "Pair"
            ]:  # do until successful, which should be soon for the way we did the coefficients
                pair1 = Pair(
                    world=self.world,
                    qubits=[self.station1.create_qubit(), self.station2.create_qubit()],
                    initial_state=initial_state,
                )
                pair2 = Pair(
                    world=self.world,
                    qubits=[self.station1.create_qubit(), self.station2.create_qubit()],
                    initial_state=initial_state,
                )
                event = EntanglementPurificationEvent(
                    time=self.world.event_queue.current_time,
                    pairs=[pair1, pair2],
                    communication_time=distance(self.station1, self.station2)
                    / _COMMUNICATION_SPEED,
                    protocol="dejmps",
                )
                self.world.event_queue.add_event(event)
                self.world.event_queue.resolve_next_event()  # resolve event
                self.world.event_queue.resolve_next_event()  # resolve unblocking/discarding generated by the event
            self.assertEqual(len(self.world.world_objects["Pair"]), 1)
            pair = self.world.world_objects["Pair"][0]
            self.assertTrue(np.allclose(pair.state, trusted_state))
            # not sure how you would test success probability without making a huge number of cases

    # # test if the use of a bogus protocol works as expected
    def test_custom_epp_event(self):
        pair1 = Pair(
            world=self.world,
            qubits=[self.station1.create_qubit(), self.station2.create_qubit()],
            initial_state=np.dot(mat.phiminus, mat.H(mat.phiminus)),
        )
        pair2 = Pair(
            world=self.world,
            qubits=[self.station1.create_qubit(), self.station2.create_qubit()],
            initial_state=np.dot(mat.psiplus, mat.H(mat.psiplus)),
        )
        event = EntanglementPurificationEvent(
            time=self.world.event_queue.current_time,
            pairs=[pair1, pair2],
            communication_time=distance(self.station1, self.station2)
            / _COMMUNICATION_SPEED,
            protocol=_bogus_epp,
        )
        self.world.event_queue.add_event(event)
        self.world.event_queue.resolve_next_event()  # resolve event
        self.world.event_queue.resolve_next_event()  # resole unblocking/discarding generated by the event
        self.assertEqual(len(self.world.world_objects["Pair"]), 1)
        pair = self.world.world_objects["Pair"][0]
        self.assertTrue(
            np.allclose(pair.state, np.dot(mat.phiplus, mat.H(mat.phiplus)))
        )


class TestBlockingSystem(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.event_queue = self.world.event_queue

    def test_blocking(self):
        test_objects = [WorldObject(world=self.world) for i in range(10)]
        test_dummy_event = DummyEvent(time=0, required_objects=test_objects)
        test_dummy_event._main_effect = MagicMock(name="_main_effect")
        self.event_queue.add_event(test_dummy_event)
        self.event_queue.resolve_next_event()
        test_dummy_event._main_effect.assert_called()
        # if one of the objects is blocked, the event should not happen
        test_objects[3].is_blocked = True
        test_dummy_event = DummyEvent(time=0, required_objects=test_objects)
        test_dummy_event._main_effect = MagicMock(name="_main_effect")
        self.event_queue.add_event(test_dummy_event)
        with self.assertWarns(UserWarning):
            self.event_queue.resolve_next_event()
        test_dummy_event._main_effect.assert_not_called()
        # ... unless the event specifically is allowed to affect blocked objects
        # e.g. DiscardQubitEvent and UnblockEvent
        test_objects[3].is_blocked = True
        test_dummy_event = DummyEvent(
            time=0, required_objects=test_objects, ignore_blocked=True
        )
        test_dummy_event._main_effect = MagicMock(name="_main_effect")
        self.event_queue.add_event(test_dummy_event)
        self.event_queue.resolve_next_event()
        test_dummy_event._main_effect.assert_called()


class TestEventQueue(unittest.TestCase):
    def setUp(self):
        self.event_queue = EventQueue()

    def test_scheduling_events(self):
        dummy_event = DummyEvent(3.3)
        self.event_queue.add_event(dummy_event)
        self.assertIn(dummy_event, self.event_queue.queue)
        num_events = 30
        more_dummy_events = [DummyEvent(time=i) for i in range(num_events, 0, -1)]
        for event in more_dummy_events:
            self.event_queue.add_event(event)
        self.assertEqual(len(self.event_queue), num_events + 1)
        # trying to schedule event in the past
        with self.assertRaises(ValueError):
            self.event_queue.add_event(DummyEvent(time=-2))

    def test_resolving_events(self):
        mockEvent1 = MagicMock(time=0)
        mockEvent2 = MagicMock(time=1)
        self.event_queue.add_event(mockEvent2)
        self.event_queue.add_event(mockEvent1)  # events added to queue in wrong order
        self.event_queue.resolve_next_event()
        mockEvent1.resolve.assert_called_once()
        mockEvent2.resolve.assert_not_called()
        self.event_queue.resolve_next_event()
        mockEvent2.resolve.assert_called_once()

    def test_resolve_until(self):
        num_events = 20
        mock_events = [MagicMock(time=i) for i in range(num_events)]
        for event in mock_events:
            self.event_queue.add_event(event)
        target_time = 5
        self.event_queue.resolve_until(target_time)
        self.assertEqual(
            len(self.event_queue), num_events - (np.floor(target_time) + 1)
        )
        self.assertEqual(self.event_queue.current_time, target_time)
        with self.assertRaises(ValueError):  # if given target_time in the past
            self.event_queue.resolve_until(0)


class TestPrioritySystem(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.event_queue = self.world.event_queue

    def test_priority_sorting(self):
        test_time = np.random.random() * 20
        important_event = PriorityEvent(time=test_time, priority=0)
        normal_event = PriorityEvent(time=test_time, priority=20)
        unimportant_event = PriorityEvent(time=test_time, priority=39)
        self.event_queue.add_event(normal_event)
        self.event_queue.add_event(unimportant_event)
        self.event_queue.add_event(important_event)
        self.assertEqual(
            self.event_queue.queue, [important_event, normal_event, unimportant_event]
        )

    def test_reasonable_priorities(self):
        test_time = np.random.random() * 20
        test_qubit = Qubit(world=self.world)
        discard_qubit_event = DiscardQubitEvent(time=test_time, qubit=test_qubit)
        normal_event = DummyEvent(time=test_time)
        unblock_event = UnblockEvent(time=test_time, quantum_objects=[test_qubit])
        self.event_queue.add_event(discard_qubit_event)
        self.event_queue.add_event(normal_event)
        self.event_queue.add_event(unblock_event)
        self.assertEqual(
            self.event_queue.queue, [unblock_event, normal_event, discard_qubit_event]
        )


if __name__ == "__main__":
    unittest.main()
