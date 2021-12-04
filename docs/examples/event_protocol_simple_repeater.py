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
from requsim.tools.protocol import TwoLinkProtocol
from requsim.events import EntanglementSwappingEvent


def SimpleProtocol(TwoLinkProtocol):
    # TwoLinkProtocol is a helpful abstract class that provides useful
    # attributes and methods for dealing with and evaluating repeater protocols
    # consisting of two links (supports multi-mode memories as well).
    # It needs to be initialized via its `setup` method after world setup
    # is complete, in order to find all the relevant objects in the world.
    def check(self):
        """Check current status and schedule new events.

        This looks globally at the status of the whole `world` and decides
        which events should be scheduled because of it. It is intended to be
        called after every change in the `world`.
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
