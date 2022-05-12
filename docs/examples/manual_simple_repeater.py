"""A simple repeater protcol performed manually with requsim objects.

Obviously, this is not the recommended way to do this, but it gives insight
into the low-level steps that need to be performed by the simulation.

This Protocol distributes a Bell pair between two Stations A and B via a
repeater station in-between that performs entanglement swapping.
Note that this implementation does not concern itself with timing and
failed trials, but of course this could be added manually as well.
"""
import numpy as np
from requsim.quantum_objects import Station, Source, Pair
from requsim.world import World
import requsim.libs.matrix as mat

total_length = 2000  # meters
initial_state = mat.phiplus @ mat.H(mat.phiplus)  # perfect Bell state

# Step 1: Perform world setup
world = World()
station_a = Station(world=world, position=0, label="Alice")
station_b = Station(world=world, position=total_length, label="Bob")
station_central = Station(world=world, position=total_length / 2, label="Repeater")
# Sources generate an entangled pair and place them into storage at two stations
# here they are positioned at the central station and send qubits out to the
# outer stations
source_a = Source(
    world=world,
    position=station_central.position,
    target_stations=[station_a, station_central],
)
source_b = Source(
    world=world,
    position=station_central.position,
    target_stations=[station_central, station_b],
)
if __name__ == "__main__":
    print("World status after Step 1: Perform world setup")
    world.print_status()

# Step 2: Distribute pairs
pair_a = source_a.generate_pair(initial_state=initial_state)
pair_b = source_b.generate_pair(initial_state=initial_state)
if __name__ == "__main__":
    print("\n\nWorld status after Step 2: Distribute pairs")
    world.print_status()

# Step 3: perform entanglement swapping
four_qubit_state = mat.tensor(pair_a.state, pair_b.state)
operator = mat.tensor(mat.I(2), mat.H(mat.phiplus), mat.I(2))
state_after_swapping = operator @ four_qubit_state @ mat.H(operator)
state_after_swapping = state_after_swapping / np.trace(state_after_swapping)
# remove outdated objects
pair_a.destroy()  # destroying a Pair object does not remove the associated qubits
pair_b.destroy()
for qubit in station_central.qubits:
    qubit.destroy()
new_pair = Pair(
    world=world,
    qubits=[station_a.qubits[0], station_b.qubits[0]],
    initial_state=state_after_swapping,
)
if __name__ == "__main__":
    print("\n\nWorld status after Step 3: Perform entanglement swapping")
    world.print_status()

# at this point you would probably collect information about the long distance
# state before removing it from the world
