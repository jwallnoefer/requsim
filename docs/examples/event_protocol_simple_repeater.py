"""Using the event system to run a simple repeater protocol.

This variant accurately uses timing data and collects information about the
generated long-distance states. The protocol decisions are made at a global
level, with the protocol looking at the whole state of the world and deciding
what to do next.

The world setup and protocol can be described as follows:

Stations A and B want to establish entangled pairs via a repeater station at
the middle point between them.
The middle station has a mechanism to create entangled pairs and a quantum
memory (can hold one qubit for each side). The middle station will continously
try to establish pairs between itself and both A and B. After it has established
a pair with both A and B, an entanglement swapping operation is performed to
connect A and B. Repeat until the desired number of pairs has been established.

Error model: The entangled pairs are considered to be perfect upon generation.
The quantum memories introduce a time-dependent dephasing noise while a qubit
sits in memory. The Bell measurement to perform the entanglement swapping is
assumed to be perfect.

For a detailed explanation of this type of protocol see e.g.
D. Luong, L. Jiang, J. Kim, N. Lütkenhaus; Applied Physics B 122, 96 (2016)
Preprint: https://arxiv.org/abs/1508.02811
The protocol in this example corresponds to the 'simultaneous' variant of the
protocol discussed there, but with a simpler error model.
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


class SimpleProtocol(TwoLinkProtocol):
    # TwoLinkProtocol is a helpful abstract class that provides useful
    # attributes and methods for dealing with and evaluating repeater protocols
    # consisting of two links (supports multi-mode memories as well).
    # It needs to be initialized via its `setup` method after world setup
    # is complete, in order to find all the relevant objects in the world.
    def check(self, message=None):
        """Check current status and schedule new events.

        This looks globally at the status of the whole `world` and decides
        which events should be scheduled because of it. It is intended to be
        called after every change in the `world`.

        Parameters
        ----------
        message : dict or None
            Optional additional information for the Protocol to consider.
            This particular protocol does not use it. Default is None.

        Returns
        -------
        None

        """
        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        num_left_pairs_scheduled = len(self._left_pairs_scheduled())
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        num_right_pairs_scheduled = len(self._right_pairs_scheduled())
        long_distance_pairs = self._get_long_range_pairs()

        # STEP 1: For each link, if there are no pairs established and
        #         no pairs scheduled: Schedule a pair.
        if num_left_pairs + num_left_pairs_scheduled == 0:
            self.source_A.schedule_event()
        if num_right_pairs + num_right_pairs_scheduled == 0:
            self.source_B.schedule_event()

        # STEP 2: If both links are present, do entanglement swapping.
        if num_left_pairs == 1 and num_right_pairs == 1:
            left_pair = left_pairs[0]
            right_pair = right_pairs[0]
            ent_swap_event = EntanglementSwappingEvent(
                time=self.world.event_queue.current_time,
                pairs=[left_pair, right_pair],
                station=self.station_central,
            )
            self.world.event_queue.add_event(ent_swap_event)

        # STEP 3: If a long range pair is present, save its data and delete
        #         the associated objects.
        if long_distance_pairs:
            for pair in long_distance_pairs:
                self._eval_pair(pair)
                for qubit in pair.qubits:
                    qubit.destroy()
                pair.destroy()


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
        return time_taken

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
    protocol = SimpleProtocol(world=world, communication_speed=C)
    protocol.setup()

    current_message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(message=current_message)
        current_message = world.event_queue.resolve_next_event()
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
        columns=[
            "raw_rate",
            "fidelity",
            "fidelity_std_err",
            "key_per_time",
            "key_per_time_std_err",
        ],
    )
    print(results)
    # plotting key_per_time vs. length is usually what you want to do with these
