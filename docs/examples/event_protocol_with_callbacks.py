"""An alternative way to use the event system.

This implements the same protocol as event_protocol_simple_repeater.py with
a different mechanism: Instead of globally examining the whole status of the
simulation and deciding from there what events to add to the event_queue, the
next thing to do after a event is resolved is already attached to the event via
a callback.

While this breaks the neat interpretation of the protocol deciding what to do
next purely from the current state of the simulation, this way can save time
especially for complex setups where the current state may be hard to analyze.
"""
import numpy as np
import pandas as pd
from requsim.tools.protocol import TwoLinkProtocol
from requsim.tools.noise_channels import z_noise_channel
from requsim.tools.evaluation import standard_bipartite_evaluation
from requsim.libs.aux_functions import distance
import requsim.libs.matrix as mat
from requsim.events import EntanglementSwappingEvent
from requsim.world import World
from requsim.noise import NoiseChannel, NoiseModel
from requsim.quantum_objects import Station, SchedulingSource


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_func(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return NoiseChannel(n_qubits=1, channel_function=dephasing_noise_func)


class CallbackProtocol(TwoLinkProtocol):
    def _check_for_swapping_A(self, event_dict):
        right_pairs = self._get_right_pairs()
        if not right_pairs:
            return
        left_pair = event_dict["output_pair"]
        right_pair = right_pairs[0]
        ent_swap_event = EntanglementSwappingEvent(
            time=self.world.event_queue.current_time,
            pairs=[left_pair, right_pair],
            station=self.station_central,
        )
        ent_swap_event.add_callback(self._after_swapping)
        self.world.event_queue.add_event(ent_swap_event)

    def _check_for_swapping_B(self, event_dict):
        left_pairs = self._get_left_pairs()
        if not left_pairs:
            return
        left_pair = left_pairs[0]
        right_pair = event_dict["output_pair"]
        ent_swap_event = EntanglementSwappingEvent(
            time=self.world.event_queue.current_time,
            pairs=[left_pair, right_pair],
            station=self.station_central,
        )
        self.world.event_queue.add_event(ent_swap_event)
        ent_swap_event.add_callback(self._after_swapping)

    def _after_swapping(self, event_dict):
        long_distance_pairs = self._get_long_range_pairs()
        if long_distance_pairs:
            for pair in long_distance_pairs:
                self._eval_pair(pair)
                for qubit in pair.qubits:
                    qubit.destroy()
                pair.destroy()
        self.start()

    def start(self):
        """Start the event chain.

        This needs to be called once to schedule the initial events.
        """
        initial_event_A = self.source_A.schedule_event()
        initial_event_A.add_callback(self._check_for_swapping_A)
        initial_event_B = self.source_B.schedule_event()
        initial_event_B.add_callback(self._check_for_swapping_B)

    def check(self):
        pass


def run(length, max_iter, params):
    C = params["COMMUNICATION_SPEED"]
    P_LINK = params["P_LINK"]
    T_DP = params["T_DP"]
    LAMBDA_BSM = params["LAMBDA_BSM"]
    L_ATT = 22e3  # attenuation length

    # define functions for link generation behavior
    def state_generation(source):
        # this returns the density matrix of a successful trial
        # this particular function already assumes some things that are only
        # appropriate for this particular setup, e.g. the source is at one
        # of the stations and the end stations do not decohere
        state = mat.phiplus @ mat.H(mat.phiplus)
        comm_distance = max(
            distance(source, source.target_stations[0]),
            distance(source, source.target_stations[1]),
        )
        storage_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:
                state = station.memory_noise.apply_to(
                    rho=state, qubit_indices=[idx], t=storage_time
                )
        return state

    def time_distribution(source):
        comm_distance = max(
            distance(source, source.target_stations[0]),
            distance(source, source.target_stations[1]),
        )
        trial_time = 2 * comm_distance / C
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        num_trials = np.random.geometric(eta)
        time_taken = num_trials * trial_time
        return time_taken, num_trials

    def BSM_error_func(rho):
        return LAMBDA_BSM * rho + (1 - LAMBDA_BSM) * mat.I(4) / 4

    BSM_noise_channel = NoiseChannel(n_qubits=2, channel_function=BSM_error_func)
    BSM_noise_model = NoiseModel(channel_before=BSM_noise_channel)

    # perform the world setup
    world = World()
    station_A = Station(world=world, position=0)
    station_central = Station(
        world=world,
        position=length / 2,
        memory_noise=construct_dephasing_noise_channel(T_DP),
        BSM_noise_model=BSM_noise_model,
    )
    station_B = Station(world=world, position=length)
    source_A = SchedulingSource(
        world=world,
        position=station_central.position,
        target_stations=[station_A, station_central],
        time_distribution=time_distribution,
        state_generation=state_generation,
    )
    source_B = SchedulingSource(
        world=world,
        position=station_central.position,
        target_stations=[station_central, station_B],
        time_distribution=time_distribution,
        state_generation=state_generation,
    )
    protocol = CallbackProtocol(world=world, communication_speed=C)
    protocol.setup()
    protocol.start()

    while len(protocol.time_list) < max_iter:
        world.event_queue.resolve_next_event()
    return protocol


if __name__ == "__main__":
    params = {
        "P_LINK": 0.80,  # link generation probability
        "T_DP": 100e-3,  # dephasing time
        "LAMBDA_BSM": 0.99,  # Bell-State-Measurement ideality parameter
        "COMMUNICATION_SPEED": 2e8,  # speed of light in optical fibre
    }
    length_list = np.linspace(20e3, 200e3, num=8)
    max_iter = 1000
    raw_data = [
        run(length=length, max_iter=max_iter, params=params).data
        for length in length_list
    ]
    result_list = [standard_bipartite_evaluation(data_frame=df) for df in raw_data]
    results = pd.DataFrame(
        data=result_list,
        index=length_list,
        columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std"],
    )
    print(results)
    # plotting key_per_time vs. length is usually what you want to do with these
