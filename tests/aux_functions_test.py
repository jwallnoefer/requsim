import unittest
from unittest.mock import MagicMock, call
import numpy as np
from requsim.libs.aux_functions import (
    apply_single_qubit_map,
    w_noise_channel,
    apply_m_qubit_map,
)
import requsim.libs.matrix as mat


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
    test_state = test_state / np.trace(
        test_state
    )  # normalize, now we have random real density matrix
    return test_state


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


if __name__ == "__main__":
    unittest.main()
