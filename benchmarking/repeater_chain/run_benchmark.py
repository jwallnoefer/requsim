"""Run the defined benchmarking case on a single machine.

Usage: run_benchmark.py [--case=CASE] [--parallel=POOL] [--output=DIR]
       run_benchmark.py --plot [--data=DIR] [--name=NAME]

Options:
    -h --help           show this help message
    --case=CASE         Specify the index of the case to run. If none is specified, runs all cases.
    --parallel=POOL     This many parts are allowed to run in parallel (with multiprocessing.Pool) [default: 1]
    --output=DIR        Any directory where to put the results [default: results]
    --plot              Make plot instead of running the cases.
    --data=DIR          Directory where the results are stored.
                        Usually where --output was specified before. [default: results]
    --name=NAME         What the plot file should be called. [default: benchmark.pdf]
"""
import os
from copy import deepcopy
from multiprocessing import Pool
from time import time

import numpy as np
from docopt import docopt
from matplotlib import pyplot as plt

from repeater_chain import run
from protocols import (
    DefaultManylinkProtocol,
    ObserveOnlyManylinkProtocol,
    CustomManylinkProtocol,
    CompositeProtocol,
    LocalProtocol,
    CallbackProtocol,
)

PROTOCOLS = {
    "Default": DefaultManylinkProtocol(),
    "ObserveOnly": ObserveOnlyManylinkProtocol(),
    "ScenarioOptimized": CustomManylinkProtocol(),
    "LocalCallbacks": CompositeProtocol(subprotocol=LocalProtocol()),
    "GlobalCallbacks": CallbackProtocol(),
}

base_num_parts = 128
base_num_links = np.linspace(0, 1024, num=base_num_parts + 1, dtype=int)[1:]
base_max_iter = 5e3
base_total_length = 50000  # meters

base_params = {"T_DP": 10, "F_INIT": 0.999}

cases = []

for name, protocol in PROTOCOLS.items():
    num_link_list = base_num_links
    num_parts = base_num_parts
    if name == "ObserveOnly":
        num_parts = base_num_parts // 2
        num_link_list = np.linspace(0, 128, num=num_parts + 1, dtype=int)[1:]
    active_case = {
        "name": name,
        "index": num_link_list,
        "num_parts": num_parts,
        "parts": [
            {
                "length": base_total_length,
                "max_iter": base_max_iter,
                "params": base_params,
                "num_links": num_link_list[part],
                "protocol": deepcopy(protocol),
            }
            for part in range(num_parts)
        ],
    }
    cases.append(active_case)


def run_timed(run_dict):
    start_time = time()
    run(**run_dict)
    return time() - start_time


if __name__ == "__main__":
    args = docopt(__doc__)
    if args["--plot"]:
        data_path = args["--data"]
        for case in cases:
            try:
                y = np.loadtxt(os.path.join(data_path, case["name"] + ".txt"))
            except FileNotFoundError:
                continue
            x = case["index"]
            plt.scatter(x, y, label=case["name"])
        plt.grid()
        plt.legend()
        plt.savefig(args["--name"])
    else:
        selected_case = args["--case"]
        if selected_case is None:
            cases_to_run = cases
        else:
            case_index = int(selected_case)
            cases_to_run = [cases[case_index]]
        parallel_jobs = int(args["--parallel"])
        for case in cases_to_run:
            if parallel_jobs > 1:
                pool = Pool(parallel_jobs)
                output = pool.map(run_timed, reversed(case["parts"]), chunksize=1)
                output = list(reversed(output))
            else:
                output = [run_timed(part) for part in case["parts"]]
            output = np.array(output) / base_max_iter
            output_dir = args["--output"]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, case["name"] + ".txt")
            np.savetxt(output_path, output)
