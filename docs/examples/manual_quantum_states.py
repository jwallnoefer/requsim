"""Example for manually working with the quantum objects classes.

While manually manipulating quantum states is definitely not the focus of this
simulation package (there are better tools for that), it is nonetheless a good
starting point. Knowing how the quantum objects work is also essential for
creating custom events.
"""
import numpy as np
from requsim.world import World
from requsim.quantum_objects import Qubit, Pair
import requsim.libs.matrix as mat
from requsim.tools.noise_channels import z_noise_channel

world = World()

# without specifying any specific setup, make a Pair object with a specific
# density matrix
qubit1 = Qubit(world=world)
qubit2 = Qubit(world=world)
initial_state = mat.phiplus @ mat.H(mat.phiplus)  # Bell state
pair = Pair(world=world, qubits=[qubit1, qubit2], initial_state=initial_state)
# Tip: you can use world.print_status() to show the state of world in a
# human-readable format

# now apply some map to the state. In quantum repeater context, this is most
# often a noise channel
# a) manually change the state with matrix operations
epsilon = 0.01
noise_operator = mat.tensor(mat.Z, mat.I(2))  # z_noise on qubit1
pair.state = (1 - epsilon) * pair.state + epsilon * noise_operator @ pair.state @ mat.H(
    noise_operator
)

# b) use appropriate NoiseChannel object and Qubit methods
qubit3 = Qubit(world=world)
qubit4 = Qubit(world=world)
initial_state = mat.phiplus @ mat.H(mat.phiplus)  # Bell state
pair2 = Pair(world=world, qubits=[qubit3, qubit4], initial_state=initial_state)
# z_noise_channel imported from requsim.tools.noise_channels
qubit3.apply_noise(z_noise_channel, epsilon=0.01)
assert np.allclose(pair.state, pair2.state)

# removing objects from the world works via the destroy methods
pair.destroy()
qubit1.destroy()
qubit2.destroy()
pair2.destroy()
qubit3.destroy()
qubit4.destroy()
