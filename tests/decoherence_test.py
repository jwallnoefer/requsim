"""Test for time-based decoherence.

Since properly testing if the time-based decoherence gets applied correctly
needs many moving parts (difficult to do fine-grained unittesting), we do this
in its own file.
"""

import unittest
from requsim.world import World
from requsim.quantum_objects import Pair, Station
from requsim.tools.noise_channels import x_noise_channel, z_noise_channel
import requsim.libs.matrix as mat
import numpy as np


def _apply_z_noise_on_first_qubit(rho, epsilon):
    aux = mat.tensor(mat.Z, mat.I(2))
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(aux, rho), mat.H(aux))


def _apply_x_noise_on_second_qubit(rho, epsilon):
    aux = mat.tensor(mat.I(2), mat.X)
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(aux, rho), mat.H(aux))


def _time_based_channel(epsilon_channel, dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def time_based_channel(rho, t):
        return epsilon_channel(rho=rho, epsilon=lambda_dp(t))

    return time_based_channel


class TestDecoherence(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.event_queue = self.world.event_queue

    def test_single_station_noisy(self):
        dephasing_time = 50
        error_channel = _time_based_channel(
            epsilon_channel=x_noise_channel, dephasing_time=dephasing_time
        )
        stations = [
            Station(world=self.world, id=0, position=0, memory_noise=None),
            Station(world=self.world, id=1, position=200, memory_noise=error_channel),
        ]
        qubits = [station.create_qubit() for station in stations]
        random_state = np.random.rand(4, 4)
        random_state = (
            (random_state + mat.H(random_state)) / 2 / np.trace(random_state)
        )  # make symmetric and normalize
        pair = Pair(world=self.world, qubits=qubits, initial_state=random_state)
        # now advance time and see if state changes appropriately.
        trusted_state = random_state
        time_interval = 10
        self.event_queue.advance_time(time_interval)
        pair.update_time()
        my_epsilon = (1 - np.exp(-time_interval / dephasing_time)) / 2
        trusted_state = _apply_x_noise_on_second_qubit(
            rho=trusted_state, epsilon=my_epsilon
        )
        self.assertTrue(np.allclose(pair.state, trusted_state))
        # do so many times
        def test_again(trusted_state):
            time_interval = np.random.random() * 20
            self.event_queue.advance_time(time_interval)
            pair.update_time()
            my_epsilon = (1 - np.exp(-time_interval / dephasing_time)) / 2
            trusted_state = _apply_x_noise_on_second_qubit(
                rho=trusted_state, epsilon=my_epsilon
            )
            self.assertTrue(np.allclose(pair.state, trusted_state))
            return trusted_state

        for i in range(20):
            trusted_state = test_again(trusted_state)

    def test_both_stations_noisy(self):
        dephasing_time = 50
        error_channel1 = _time_based_channel(
            epsilon_channel=z_noise_channel, dephasing_time=dephasing_time
        )
        error_channel2 = _time_based_channel(
            epsilon_channel=x_noise_channel, dephasing_time=dephasing_time
        )
        stations = [
            Station(world=self.world, id=0, position=0, memory_noise=error_channel1),
            Station(world=self.world, id=1, position=200, memory_noise=error_channel2),
        ]
        qubits = [station.create_qubit() for station in stations]
        random_state = np.random.rand(4, 4)
        random_state = (
            (random_state + mat.H(random_state)) / 2 / np.trace(random_state)
        )  # make symmetric and normalize
        pair = Pair(world=self.world, qubits=qubits, initial_state=random_state)
        # now advance time and see if state changes appropriately.
        trusted_state = random_state
        time_interval = 10
        self.event_queue.advance_time(time_interval)
        pair.update_time()
        my_epsilon = (1 - np.exp(-time_interval / dephasing_time)) / 2
        trusted_state = _apply_z_noise_on_first_qubit(
            rho=trusted_state, epsilon=my_epsilon
        )
        trusted_state = _apply_x_noise_on_second_qubit(
            rho=trusted_state, epsilon=my_epsilon
        )
        self.assertTrue(np.allclose(pair.state, trusted_state))
        # do so many times
        def test_again(trusted_state):
            time_interval = np.random.random() * 20
            self.event_queue.advance_time(time_interval)
            pair.update_time()
            my_epsilon = (1 - np.exp(-time_interval / dephasing_time)) / 2
            trusted_state = _apply_z_noise_on_first_qubit(
                rho=trusted_state, epsilon=my_epsilon
            )
            trusted_state = _apply_x_noise_on_second_qubit(
                rho=trusted_state, epsilon=my_epsilon
            )
            self.assertTrue(np.allclose(pair.state, trusted_state))
            return trusted_state

        for i in range(20):
            trusted_state = test_again(trusted_state)

    # def test_multiple_pairs(self):
    #     # this is to test that things are getting resolved correctly if
    #     # update times are staggered as they will be for long repeater chains
    #     pass


if __name__ == "__main__":
    unittest.main()
