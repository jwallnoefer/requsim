"""Tests for the MultiQubit class

Make sure this representation of multipartite states works as expected.
"""

import pytest
from requsim.world import World
import numpy as np
from requsim.quantum_objects import Qubit, MultiQubit, WorldObject
from requsim.noise import NoiseChannel
import requsim.libs.matrix as mat
import itertools as it


def _random_n_qubit_state(n):
    state = np.random.random((2**n, 2**n))
    state = (state + mat.H(state)) / 2  # symmetrize
    state = state / np.trace(state)  # normalize
    return state


def _random_pauli_diagonal_channel():
    epsilons = np.random.random(4)
    epsilons = epsilons / np.sum(epsilons)  # normalize
    e_0, e_x, e_y, e_z = epsilons

    def new_noise_function(rho):
        return (
            e_0 * rho
            + e_x * (mat.X @ rho @ mat.H(mat.X))
            + e_y * (mat.Y @ rho @ mat.H(mat.Y))
            + e_z * (mat.Z @ rho @ mat.H(mat.Z))
        )

    return NoiseChannel(n_qubits=1, channel_function=new_noise_function)


@pytest.fixture
def world():
    return World()


def test_world_object_general(world):
    # make sure MultiQubit has the behavior expected from a WorldObject
    for num_qubits in range(2, 11):
        test_object = MultiQubit(
            world=world,
            qubits=[Qubit(world=world) for i in range(num_qubits)],
            initial_state=_random_n_qubit_state(num_qubits),
        )
        assert isinstance(test_object, WorldObject)
        assert test_object.world is world
        assert test_object.event_queue is world.event_queue
        assert test_object in world
        assert test_object in world.world_objects[test_object.type]
        world.event_queue.current_time = np.random.random()
        test_object.update_time()
        assert world.event_queue.current_time == test_object.last_updated
        # check that str and rerpr do not cause any errors
        print(str(test_object))
        repr(test_object)
        # does destroying work?
        test_object.destroy()
        assert test_object not in world
        assert test_object not in world.world_objects[test_object.type]


def test_multiple_instances(world):
    test_objects = []
    for num_qubits in range(2, 11):
        test_objects += [
            MultiQubit(
                world=world,
                qubits=[Qubit(world=world) for i in range(num_qubits)],
                initial_state=_random_n_qubit_state(num_qubits),
            )
        ]
        test_objects += [
            MultiQubit(
                world=world,
                qubits=[Qubit(world=world) for i in range(num_qubits)],
                initial_state=_random_n_qubit_state(num_qubits),
            )
        ]
    for a, b in it.combinations(test_objects, 2):
        assert a.label != b.label


def test_basic_properties(world):
    for num_qubits in range(2, 11):
        initial_state = _random_n_qubit_state(num_qubits)
        qubits = [Qubit(world=world) for i in range(num_qubits)]
        test_object = MultiQubit(
            world=world, qubits=qubits, initial_state=initial_state
        )
        for qubit in qubits:
            assert qubit in test_object.qubits
        for qubit in test_object.qubits:
            assert qubit in qubits
        assert test_object.num_qubits == len(test_object.qubits)
        assert np.allclose(test_object.state, initial_state)


def test_properties_are_readonly(world):
    for num_qubits in range(2, 11):
        test_object = MultiQubit(
            world=world,
            qubits=[Qubit(world=world) for i in range(num_qubits)],
            initial_state=_random_n_qubit_state(num_qubits),
        )
        hash(test_object.type)
        with pytest.raises(AttributeError):
            test_object.type = "some string"
        hash(test_object.num_qubits)
        with pytest.raises(AttributeError):
            test_object.num_qubits = 42
        hash(test_object.qubits)
        with pytest.raises(AttributeError):
            test_object.qubits = ("some", "fantasy", "object")


def test_update_times(world):
    for num_qubits in range(2, 11):
        test_object = MultiQubit(
            world=world,
            qubits=[Qubit(world=world) for i in range(num_qubits)],
            initial_state=_random_n_qubit_state(num_qubits),
        )
        world.event_queue.current_time = np.random.random()
        test_object.update_time()
        for qubit in test_object.qubits:
            assert qubit.last_updated == world.event_queue.current_time


def test_noise_handling(world):
    # so far the delegation system only passes noise up from the qubit level,
    # everything else is handled directly at the state level, but that could
    # change in the future
    for num_qubits in range(2, 7):
        initial_state = _random_n_qubit_state(num_qubits)
        compare_state = np.copy(initial_state)
        test_object = MultiQubit(
            world=world,
            qubits=[Qubit(world=world) for i in range(num_qubits)],
            initial_state=initial_state,
        )
        noise_channels = [_random_pauli_diagonal_channel() for i in range(num_qubits)]
        for qubit, noise_channel in zip(test_object.qubits, noise_channels):
            qubit.apply_noise(noise_channel)
        for qubit_index, noise_channel in enumerate(noise_channels):
            compare_state = noise_channel.apply_to(
                rho=compare_state, qubit_indices=[qubit_index]
            )
        assert np.allclose(test_object.state, compare_state)
    # now do the same, but with noise before the MultiQubit is instantiated
    for num_qubits in range(2, 7):
        initial_state = _random_n_qubit_state(num_qubits)
        compare_state = np.copy(initial_state)
        qubits = [Qubit(world=world) for i in range(num_qubits)]
        noise_channels = [_random_pauli_diagonal_channel() for i in range(num_qubits)]
        for qubit, noise_channel in zip(qubits, noise_channels):
            qubit.apply_noise(noise_channel)
        for qubit_index, noise_channel in enumerate(noise_channels):
            compare_state = noise_channel.apply_to(
                rho=compare_state, qubit_indices=[qubit_index]
            )
        test_object = MultiQubit(
            world=world, qubits=qubits, initial_state=initial_state
        )
        assert np.allclose(test_object.state, compare_state)
