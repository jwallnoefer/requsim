"""Test of the internal workings of the noise system.

While a lot of this is already covered by tests of the quantum_objects and
time-based decoherence, this file adds tests to make sure it works for arbitrary
noise channels and the internal delegation mechanisms (like unresolved noiseperform as expected.
"""
import pytest
import numpy as np
from requsim.world import World
from requsim.quantum_objects import Qubit
from requsim.noise import NoiseChannel
import requsim.libs.matrix as mat
from unittest.mock import MagicMock


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


def _generate_mock_channel():
    mock = MagicMock()
    return NoiseChannel(n_qubits=1, channel_function=mock), mock


def _always_fail_handler(noise_channel, *args, **kwargs):
    handling_successful = False
    return handling_successful


def _always_succeed_handler(noise_channel, *args, **kwargs):
    handling_successful = True
    return handling_successful


@pytest.fixture
def world():
    return World()


def test_manual_delegation(world):
    # what if an arbitrary noise handling function is specified?
    qubit = Qubit(world=world)
    test_channel = _random_pauli_diagonal_channel()
    qubit.add_noise_handler(_always_fail_handler)
    qubit.apply_noise(test_channel)
    # noise handling should have failed and therefore is still unresolved
    assert len(qubit._unresolved_noises) == 1
    qubit.add_noise_handler(_always_succeed_handler)
    # now finally noise should have been handled
    assert len(qubit._unresolved_noises) == 0


def test_manual_delegation_of_frozen_channel(world):
    qubit = Qubit(world=world)
    test_channel, mock_func = _generate_mock_channel()
    qubit.apply_noise(
        test_channel, "myarg1", "myarg2", mykwarg1="mykwarg1", mykwarg2="mykwarg2"
    )
    # noise could not be handled
    assert len(qubit._unresolved_noises) == 1
    mock_rho = MagicMock()

    def custom_handler(noise_channel):
        noise_channel(mock_rho)
        handling_successful = True
        return handling_successful

    qubit.add_noise_handler(custom_handler)
    # noise has been handled
    assert len(qubit._unresolved_noises) == 0
    # handling has been done by calling the mock_func appropriately
    mock_func.assert_called_with(
        mock_rho, "myarg1", "myarg2", mykwarg1="mykwarg1", mykwarg2="mykwarg2"
    )
