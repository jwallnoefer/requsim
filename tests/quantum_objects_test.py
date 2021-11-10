import unittest
from unittest.mock import MagicMock
from requsim.world import World
from requsim.quantum_objects import (
    Qubit,
    Pair,
    Station,
    Source,
    SchedulingSource,
    WorldObject,
)
from requsim.events import SourceEvent, EntanglementSwappingEvent
import numpy as np
import requsim.libs.matrix as mat
from requsim.noise import NoiseChannel, NoiseModel
from requsim.libs.aux_functions import apply_single_qubit_map
from requsim.tools.noise_channels import w_noise_channel

example_noise_channel = NoiseChannel(
    n_qubits=1, channel_function=lambda rho: w_noise_channel(rho=rho, alpha=0.99)
)

example_four_qubit_channel = NoiseChannel(
    n_qubits=4,
    channel_function=lambda rho: example_noise_channel.apply_to(
        example_noise_channel.apply_to(rho=rho, qubit_indices=[1]), qubit_indices=[2]
    ),
)
example_two_qubit_channel = NoiseChannel(
    n_qubits=2,
    channel_function=lambda rho: example_noise_channel.apply_to(
        example_noise_channel.apply_to(rho=rho, qubit_indices=[0]), qubit_indices=[1]
    ),
)


def _epp_map_dummy_replace(rho):
    return mat.phiplus @ mat.H(mat.phiplus)


class TestQuantumObjects(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.event_queue = self.world.event_queue

    def _aux_general_test(self, quantum_object):
        self.assertIsInstance(quantum_object, WorldObject)
        # assert that quantum object is registered
        self.assertIn(quantum_object, self.world.world_objects[quantum_object.type])
        # or do the same thing in the new syntax way
        self.assertIn(quantum_object, self.world)
        # assert that object is in test world and has correct event queue
        self.assertIs(quantum_object.world, self.world)
        self.assertIs(quantum_object.event_queue, self.world.event_queue)
        # see if updating time works as expected
        self.world.event_queue.current_time = 1  # artificially advance time 1 second
        quantum_object.update_time()
        self.assertEqual(
            quantum_object.last_updated, self.world.event_queue.current_time
        )

    def test_qubit(self):
        qubit = Qubit(world=self.world, station=MagicMock(memory_noise=None))
        self._aux_general_test(qubit)

    def test_pair(self):
        qubits = [
            Qubit(world=self.world, station=MagicMock(memory_noise=None))
            for i in range(2)
        ]
        pair = Pair(
            world=self.world, qubits=qubits, initial_state=np.diag([1, 0, 0, 0])
        )
        self._aux_general_test(pair)

    def test_unresolved_noise(self):
        noisy_qubit = Qubit(
            world=self.world,
            station=MagicMock(None),
            unresolved_noise=example_noise_channel,
        )
        other_qubit = Qubit(world=self.world, station=MagicMock(None))
        test_state = np.random.random((4, 4))
        test_state = 1 / 2 * (test_state + mat.H(test_state))
        test_state = test_state / np.trace(test_state)
        pair = Pair(
            world=self.world,
            qubits=[noisy_qubit, other_qubit],
            initial_state=test_state,
        )
        expected_state = apply_single_qubit_map(
            map_func=lambda mu: w_noise_channel(rho=mu, alpha=0.99),
            qubit_index=0,
            rho=test_state,
        )
        self.assertTrue(np.allclose(pair.state, expected_state))

    def test_station(self):
        station = Station(world=self.world, id=1, position=0)
        self._aux_general_test(station)
        qubit = station.create_qubit()
        self.assertIsInstance(qubit, Qubit)
        self.assertIn(qubit, station.qubits)
        self.assertIs(qubit.station, station)
        # now test if destroying the qubit properly deregisters it
        qubit.destroy()
        self.assertNotIn(qubit, station.qubits)

    def test_cutoff_time_station(self):
        cutoff_time = np.random.random() * 40
        station = Station(
            world=self.world, id=1, position=0, memory_cutoff_time=cutoff_time
        )
        qubit = station.create_qubit()
        self.assertIn(qubit, self.world.world_objects[qubit.type])
        self.event_queue.resolve_until(cutoff_time - 10 ** -6 * cutoff_time)
        self.assertIn(qubit, self.world.world_objects[qubit.type])
        self.event_queue.resolve_until(cutoff_time)
        self.assertNotIn(qubit, self.world.world_objects[qubit.type])

    def test_station_options(self):
        # test dark_count_probability
        station = Station(world=self.world, position=0)  # not specified
        self.assertEqual(station.dark_count_probability, 0)
        random_num = np.random.random()
        station = Station(
            world=self.world, position=0, dark_count_probability=random_num
        )
        self.assertEqual(station.dark_count_probability, random_num)
        # test map on creation (which is used for alignment error)
        station = Station(
            world=self.world, position=0, creation_noise_channel=example_noise_channel
        )
        test_state = np.random.random((4, 4))
        test_state = 1 / 2 * (test_state + mat.H(test_state))
        test_state = test_state / np.trace(test_state)
        station_qubit = station.create_qubit()
        other_qubit = Qubit(world=self.world, station=MagicMock(None))
        pair = Pair(
            world=self.world,
            qubits=[station_qubit, other_qubit],
            initial_state=test_state,
        )
        expected_state = apply_single_qubit_map(
            map_func=lambda mu: w_noise_channel(rho=mu, alpha=0.99),
            qubit_index=0,
            rho=test_state,
        )
        self.assertTrue(np.allclose(pair.state, expected_state))

    def test_BSM_station_option(self):
        # # test BSM_noise_model option (which is used by entanglement swapping)
        # first try the two options for before-noise and see if they give the same result as expected
        test_state = np.random.random((4, 4))
        test_state = 1 / 2 * (test_state + mat.H(test_state))
        test_state = test_state / np.trace(test_state)
        left_station = Station(world=self.world, position=0)
        right_station = Station(world=self.world, position=0)
        four_qubit_noise_model = NoiseModel(channel_before=example_four_qubit_channel)
        station_a = Station(
            world=self.world, position=0, BSM_noise_model=four_qubit_noise_model
        )
        pair_1a = Pair(
            world=self.world,
            qubits=[left_station.create_qubit(), station_a.create_qubit()],
            initial_state=test_state,
        )
        pair_2a = Pair(
            world=self.world,
            qubits=[station_a.create_qubit(), right_station.create_qubit()],
            initial_state=test_state,
        )
        event_a = EntanglementSwappingEvent(time=0, pairs=[pair_1a, pair_2a])
        self.world.event_queue.add_event(event_a)
        self.world.event_queue.resolve_next_event()
        pair_a = self.world.world_objects["Pair"][0]
        two_qubit_noise_model = NoiseModel(channel_before=example_two_qubit_channel)
        station_b = Station(
            world=self.world, position=0, BSM_noise_model=two_qubit_noise_model
        )
        pair_1b = Pair(
            world=self.world,
            qubits=[left_station.create_qubit(), station_b.create_qubit()],
            initial_state=test_state,
        )
        pair_2b = Pair(
            world=self.world,
            qubits=[station_b.create_qubit(), right_station.create_qubit()],
            initial_state=test_state,
        )
        event_b = EntanglementSwappingEvent(time=0, pairs=[pair_1b, pair_2b])
        self.world.event_queue.add_event(event_b)
        self.world.event_queue.resolve_next_event()
        pair_b = self.world.world_objects["Pair"][1]
        self.assertTrue(np.allclose(pair_a.state, pair_b.state))
        # try the replace option
        replace_noise_model = NoiseModel(map_replace=_epp_map_dummy_replace)
        station_c = Station(
            world=self.world, position=0, BSM_noise_model=replace_noise_model
        )
        pair_1c = Pair(
            world=self.world,
            qubits=[left_station.create_qubit(), station_c.create_qubit()],
            initial_state=test_state,
        )
        pair_2c = Pair(
            world=self.world,
            qubits=[station_c.create_qubit(), right_station.create_qubit()],
            initial_state=test_state,
        )
        event_c = EntanglementSwappingEvent(time=0, pairs=[pair_1c, pair_2c])
        self.world.event_queue.add_event(event_c)
        self.world.event_queue.resolve_next_event()
        pair_c = self.world.world_objects["Pair"][2]
        self.assertTrue(np.allclose(pair_c.state, mat.phiplus @ mat.H(mat.phiplus)))
        # try the after option
        after_noise_model = NoiseModel(channel_after=example_two_qubit_channel)
        station_d = Station(
            world=self.world, position=0, BSM_noise_model=after_noise_model
        )
        pair_1d = Pair(
            world=self.world,
            qubits=[left_station.create_qubit(), station_d.create_qubit()],
            initial_state=test_state,
        )
        pair_2d = Pair(
            world=self.world,
            qubits=[station_d.create_qubit(), right_station.create_qubit()],
            initial_state=test_state,
        )
        event_d = EntanglementSwappingEvent(time=0, pairs=[pair_1d, pair_2d])
        self.world.event_queue.add_event(event_d)
        self.world.event_queue.resolve_next_event()
        pair_d = self.world.world_objects["Pair"][3]
        # not sure what should be tested here... but at least it should run through
        self.assertIn(pair_d, self.world)

    def test_source(self):
        stations = [Station(world=self.world, id=i, position=200 * i) for i in range(2)]
        source = Source(world=self.world, position=100, target_stations=stations)
        self._aux_general_test(source)
        test_state = np.random.rand(4, 4)
        pair = source.generate_pair(test_state)
        self.assertIsInstance(pair, Pair)
        self.assertTrue(np.allclose(pair.state, test_state))
        pair_stations = [qubit.station for qubit in pair.qubits]
        self.assertEqual(pair_stations, stations)

    def test_scheduling_source(self):
        def dummy_schedule(source):
            return 5, 0

        def dummy_generation(source):
            return np.dot(mat.phiplus, mat.H(mat.phiplus))

        stations = [Station(world=self.world, id=i, position=200 * i) for i in range(2)]
        source = SchedulingSource(
            world=self.world,
            position=100,
            target_stations=stations,
            time_distribution=dummy_schedule,
            state_generation=dummy_generation,
        )
        self._aux_general_test(source)
        start_time = self.world.event_queue.current_time
        # now schedule an event
        source.schedule_event()
        event = self.world.event_queue.next_event
        self.assertIsInstance(event, SourceEvent)
        self.assertEqual(event.time, start_time + 5)
        self.world.event_queue.resolve_next_event()
        pair = self.world.world_objects["Pair"][0]
        self.assertTrue(
            np.allclose(pair.state, np.dot(mat.phiplus, mat.H(mat.phiplus)))
        )
        source.schedule_event()
        event = self.world.event_queue.next_event
        self.assertIsInstance(event, SourceEvent)
        self.assertEqual(event.time, start_time + 10)


if __name__ == "__main__":
    unittest.main()
