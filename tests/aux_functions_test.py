import unittest
from unittest.mock import MagicMock
import pytest
import numpy as np
from requsim.libs.aux_functions import (
    apply_single_qubit_map,
    apply_m_qubit_map,
    distance,
)
from requsim.tools.noise_channels import w_noise_channel
import requsim.libs.matrix as mat
from requsim.quantum_objects import Source, Station, WorldObject
from requsim.world import World


def _single_qubit_wnoise(rho, p):
    return (p + (1 - p) / 4) * rho + (1 - p) / 4 * (
        np.dot(np.dot(mat.X, rho), mat.H(mat.X))
        + np.dot(np.dot(mat.Y, rho), mat.H(mat.Y))
        + np.dot(np.dot(mat.Z, rho), mat.H(mat.Z))
    )


def _two_qubit_local_wnoise(rho, p):
    rho = apply_single_qubit_map(
        map_func=_single_qubit_wnoise, qubit_index=0, rho=rho, p=p
    )
    rho = apply_single_qubit_map(
        map_func=_single_qubit_wnoise, qubit_index=1, rho=rho, p=p
    )
    return rho


def _random_test_state(n):
    test_state = np.random.random((2 ** n, 2 ** n))
    test_state = test_state + test_state.T  # symmetrize
    # normalize, so we have random real density matrix
    test_state = test_state / np.trace(test_state)
    return test_state


class _PositionWorldObject(WorldObject):
    def __init__(self, world, position, label=None):
        self.position = position
        super(_PositionWorldObject, self).__init__(world=world, label=label)


class TestAuxFunctions(unittest.TestCase):
    def test_apply_single_qubit_map(self):
        # test that parameters are passed through correctly
        single_qubit_array = np.zeros((2, 2), dtype=float)
        test_func = MagicMock(return_value=single_qubit_array)
        rho = np.zeros((4, 4), dtype=float)
        apply_single_qubit_map(
            test_func, 0, rho, 3, 7, k=30, my_other_keyword="da_string"
        )
        self.assertEqual(test_func.call_args[0][1:], (3, 7))
        self.assertEqual(
            test_func.call_args[1], {"k": 30, "my_other_keyword": "da_string"}
        )
        self.assertTrue(np.all(test_func.call_args[0][0] == single_qubit_array))
        # # test that it does what we think it does
        test_state = _random_test_state(n=4).astype(complex)
        qubit_index = 0
        p = 0.9
        trusted_way = mat.wnoise(rho=test_state, n=qubit_index, p=p)
        new_way = apply_single_qubit_map(
            map_func=_single_qubit_wnoise, qubit_index=qubit_index, rho=test_state, p=p
        )
        self.assertTrue(np.allclose(trusted_way, new_way))
        new_way_again = apply_single_qubit_map(
            map_func=w_noise_channel, qubit_index=qubit_index, rho=test_state, alpha=p
        )
        self.assertTrue(np.allclose(trusted_way, new_way_again))

    def test_apply_m_qubit_map(self):
        # test that parameters are passed through correctly
        two_qubit_array = np.zeros((4, 4), dtype=float)
        test_func = MagicMock(return_value=two_qubit_array)
        rho = np.zeros((8, 8), dtype=float)
        apply_m_qubit_map(
            test_func, [0, 1], rho, 3, 7, k=30, my_other_keyword="da_string"
        )
        self.assertEqual(test_func.call_args[0][1:], (3, 7))
        self.assertEqual(
            test_func.call_args[1], {"k": 30, "my_other_keyword": "da_string"}
        )
        self.assertTrue(np.all(test_func.call_args[0][0] == two_qubit_array))
        # test with known example
        test_state = _random_test_state(n=6).astype(complex)
        p = 0.9
        qubit_indices = [1, 3]
        trusted_way = mat.wnoise(
            rho=mat.wnoise(rho=test_state, n=qubit_indices[0], p=p),
            n=qubit_indices[1],
            p=p,
        )
        new_way = apply_m_qubit_map(
            map_func=_two_qubit_local_wnoise,
            qubit_indices=qubit_indices,
            rho=test_state,
            p=p,
        )
        self.assertTrue(np.allclose(trusted_way, new_way))


# test distances with pytest instead of unittest now
def test_distance_numbers():
    assert distance(3, 4.5) == 1.5
    assert distance(-4, 2) == 6
    assert distance(2.0, -4) == 6
    for i in range(100):
        a = np.random.random()
        b = np.random.random()
        assert distance(a, b) == pytest.approx(np.abs(a - b))


def test_distance_arrays():
    assert distance(np.array([0, 1]), np.array([0, 2])) == pytest.approx(1)
    assert distance(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5)
    for i in range(100):
        num_dim = int(np.ceil(np.random.random() * 100))
        a = np.random.random(num_dim) * 65
        b = np.random.random(num_dim) * 115
        assert distance(a, b) == pytest.approx(np.sqrt(np.sum((a - b) ** 2)))


def test_distance_world_objects():
    world = World()
    position1 = 40
    position2 = 22
    a = _PositionWorldObject(world=world, position=position1)
    b = _PositionWorldObject(world=world, position=position2)
    assert distance(a, b) == pytest.approx(np.abs(position1 - position2))
    position1 = np.random.random(2) * 40
    position2 = np.random.random(2) * 22
    a = _PositionWorldObject(world=world, position=position1)
    b = _PositionWorldObject(world=world, position=position2)
    assert distance(a, b) == pytest.approx(
        np.sqrt(np.sum((position1 - position2) ** 2))
    )
    position1 = np.random.random(2) * 40
    position2 = np.random.random(2) * 22
    a = Station(world=world, position=position1)
    b = Source(world=world, position=position2, target_stations=[a, a])
    assert distance(a, b) == pytest.approx(
        np.sqrt(np.sum((position1 - position2) ** 2))
    )


def test_distance_mixed_types():
    world = World()
    a = 3
    b = _PositionWorldObject(world=world, position=5)
    assert distance(a, b) == 2
    a = _PositionWorldObject(world=world, position=np.array([0, 3.0]))
    b = np.array([4.0, 0.0])
    assert distance(a, b) == pytest.approx(5)
    a = 3
    b = np.random.random(2) * 22
    with pytest.raises(TypeError):
        distance(a, b)
    a = np.random.random(2) * 22
    b = np.random.random(3) * 23
    with pytest.raises(ValueError):
        distance(a, b)
    a = list(np.random.random(2) * 22)
    b = np.random.random(3) * 23
    with pytest.raises(TypeError):
        distance(a, b)


if __name__ == "__main__":
    unittest.main()
