"""Tests for noise channels."""
import pytest
from requsim.noise import NoiseChannel
import numpy as np
import requsim.libs.matrix as mat
from unittest.mock import MagicMock


def _random_n_qubit_state(n):
    state = np.random.random((2**n, 2**n))
    state = (state + mat.H(state)) / 2  # symmetrize
    state = state / np.trace(state)  # normalize
    return state


def _random_pauli_diagonal_noise_function():
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

    return new_noise_function


def test_call_apply_to():
    # using __call__ and apply_to should have same effect on an appropriately
    # sized density matrix
    for i in range(100):
        test_state = _random_n_qubit_state(n=1)
        test_function = _random_pauli_diagonal_noise_function()
        test_channel = NoiseChannel(n_qubits=1, channel_function=test_function)
        out0 = test_function(test_state)
        out1 = test_channel(test_state)
        assert np.allclose(out0, out1)
        out2 = test_channel.apply_to(rho=test_state, qubit_indices=[0])
        assert np.allclose(out0, out2)
        assert np.allclose(out1, out2)


def test_channel_composition():
    # local channels can easily be combined to a larger channel
    for n_qubits in range(1, 7):
        test_state = _random_n_qubit_state(n=n_qubits)
        local_channels = [
            NoiseChannel(
                n_qubits=1, channel_function=_random_pauli_diagonal_noise_function()
            )
            for i in range(n_qubits)
        ]

        def new_func(rho):
            for idx, channel in enumerate(local_channels):
                rho = channel.apply_to(rho=rho, qubit_indices=[idx])
            return rho

        test_channel = NoiseChannel(n_qubits=n_qubits, channel_function=new_func)
        out1 = test_channel(test_state)
        out2 = test_channel.apply_to(rho=test_state, qubit_indices=np.arange(n_qubits))
        assert np.allclose(out1, out2)


def _assert_called_with_without_rho(mock, *args, **kwargs):
    for call in mock.call_args_list:
        assert call.args[1:] == args
        assert call.kwargs == kwargs


def test_freeze_noise_channel():
    for n_qubits in range(1, 7):
        test_state = _random_n_qubit_state(n=n_qubits)
        for noise_qubits in range(1, n_qubits + 1):
            mock_function = MagicMock(
                return_value=_random_n_qubit_state(n=noise_qubits)
            )
            test_channel = NoiseChannel(
                n_qubits=noise_qubits, channel_function=mock_function
            )
            # without frozen args/kwargs
            frozen_channel = test_channel.freeze()
            if n_qubits == noise_qubits:
                frozen_channel(test_state)
                mock_function.assert_called_with(test_state)
                mock_function.reset_mock()
            qubit_indices = np.random.choice(np.arange(n_qubits), size=noise_qubits)
            frozen_channel.apply_to(test_state, qubit_indices=qubit_indices)
            _assert_called_with_without_rho(mock_function)
            mock_function.reset_mock()
            # with frozen args and kwargs
            test_args = (3, "bear", 42.2)
            test_kwargs = {"foo": 33, "bar": "apple"}
            frozen_channel = test_channel.freeze(*test_args, **test_kwargs)
            if n_qubits == noise_qubits:
                frozen_channel(test_state)
                mock_function.assert_called_with(test_state, *test_args, **test_kwargs)
                mock_function.reset_mock()
            qubit_indices = np.random.choice(np.arange(n_qubits), size=noise_qubits)
            frozen_channel.apply_to(test_state, qubit_indices=qubit_indices)
            _assert_called_with_without_rho(mock_function, *test_args, **test_kwargs)
            mock_function.reset_mock()
