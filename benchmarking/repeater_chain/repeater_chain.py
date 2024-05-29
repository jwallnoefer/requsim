"""Benchmarking for a simple scenario with only some basic parameters."""
from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
from functools import lru_cache
from requsim.world import World
from requsim.quantum_objects import Station, SchedulingSource
from requsim.noise import NoiseChannel
from warnings import warn
from requsim.libs.aux_functions import apply_single_qubit_map, distance
from requsim.tools.noise_channels import (
    y_noise_channel,
    z_noise_channel,
    w_noise_channel,
)
import requsim.libs.matrix as mat

from protocols import (
    DefaultManylinkProtocol,
    ObserveOnlyManylinkProtocol,
    CustomManylinkProtocol,
    CompositeProtocol,
    LocalProtocol,
    CallbackProtocol,
)

C = 2e8  # speed of light in optical fiber
L_ATT = 22000  # Attenuation length in optical fiber


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return NoiseChannel(n_qubits=1, channel_function=dephasing_noise_channel)


def construct_y_noise_channel(epsilon):
    return lambda rho: y_noise_channel(rho=rho, epsilon=epsilon)


def construct_w_noise_channel(epsilon):
    return lambda rho: w_noise_channel(rho=rho, alpha=(1 - epsilon))


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d) ** 2)


def run(length, max_iter, params, num_links, protocol):
    assert num_links % 2 == 0
    allowed_params = ["F_INIT", "T_DP"]
    P_LINK = 1
    P_D = 0
    T_P = 0
    for key in params:
        if key not in allowed_params:
            warn(f"params[{key}] is not a supported parameter and will be ignored.")
    # unpack the parameters
    F_INIT = params.get("F_INIT", 1.0)  # initial fidelity of created pairs
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError as e:
        raise Exception('params["T_DP"] is a mandatory argument').with_traceback(
            e.__traceback__
        )

    def time_distribution(source):
        comm_distance = np.max(
            [
                np.abs(source.position - source.target_stations[0].position),
                np.abs(source.position - source.target_stations[1].position),
            ]
        )
        comm_time = 2 * comm_distance / C
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - P_D) ** 2
        trial_time = (
            T_P + comm_time
        )  # I don't think that paper uses latency time and loading time?
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time

    @lru_cache()
    def state_generation(source):
        state = F_INIT * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - F_INIT) / 3 * (
            mat.psiplus @ mat.H(mat.psiplus)
            + mat.phiminus @ mat.H(mat.phiminus)
            + mat.psiminus @ mat.H(mat.psiminus)
        )
        comm_distance = np.max(
            [
                distance(source, source.target_stations[0]),
                distance(source.target_stations[1], source),
            ]
        )
        trial_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if (
                station.memory_noise is not None
            ):  # dephasing that has accrued while other qubit was travelling
                storage_time = (
                    trial_time - distance(source, station) / C
                )  # qubit is in storage for varying amounts of time
                state = apply_single_qubit_map(
                    map_func=station.memory_noise,
                    qubit_index=idx,
                    rho=state,
                    t=storage_time,
                )
        return state

    station_positions = [x * length / num_links for x in range(num_links + 1)]

    world = World()
    station_A = Station(world, position=station_positions[0], memory_noise=None)
    other_stations = [
        Station(
            world,
            position=pos,
            memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP),
        )
        for pos in station_positions[1:-1]
    ]
    station_B = Station(world, position=station_positions[-1], memory_noise=None)
    stations = [station_A] + other_stations + [station_B]
    source_positions = [station_positions[2 * (i // 2) + 1] for i in range(num_links)]
    sources = []
    for i, source_position in enumerate(source_positions):
        sources += [
            SchedulingSource(
                world,
                position=source_position,
                target_stations=(stations[i], stations[i + 1]),
                time_distribution=time_distribution,
                state_generation=state_generation,
            )
        ]
    protocol.setup(world, stations, sources, num_memories=1, communication_speed=C)

    # from code import interact
    # interact(local=locals())
    current_message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(current_message)
        try:
            current_message = world.event_queue.resolve_next_event()
        except IndexError as e:
            world.print_status()
            raise e
            from code import interact

            interact(local=locals())

    return protocol


if __name__ == "__main__":
    protocol_collection = {
        "Default": DefaultManylinkProtocol(),
        "ObserveOnly": ObserveOnlyManylinkProtocol(),
        "Custom": CustomManylinkProtocol(),
        "Local": CompositeProtocol(subprotocol=LocalProtocol()),
        "Callback": CallbackProtocol(),
    }

    num_parts = 16
    num_links = np.linspace(0, 1024, num=num_parts + 1, dtype=int)[1:]
    max_iter = 10
    output = defaultdict(list)
    total_length = 50000
    base_params = {"T_DP": 25, "F_INIT": 0.999}

    for name, protocol in protocol_collection.items():
        print(name)
        num_link_list = num_links
        if name == "ObserveOnly":
            num_link_list = np.linspace(0, 128, num=num_parts // 2 + 1, dtype=int)[1:]
        for n_links in num_link_list:
            active_protocol = deepcopy(protocol)
            start_time = time()
            res = run(
                length=total_length,
                max_iter=max_iter,
                params=base_params,
                num_links=n_links,
                protocol=active_protocol,
            )
            time_per_pair = (time() - start_time) / max_iter
            print(n_links, time_per_pair)
            output[name].append(time_per_pair)

    import matplotlib.pyplot as plt

    for name, times in output.items():
        x = num_links
        if name == "ObserveOnly":
            x = np.linspace(0, 128, num=num_parts // 2 + 1, dtype=int)[1:]
        plt.plot(x, times, label=name)
    plt.grid()
    plt.legend()
    plt.xlabel("Number of repeater links")
    plt.ylabel("Run time per pair")
    plt.savefig("benchmark.png")
    plt.show()
