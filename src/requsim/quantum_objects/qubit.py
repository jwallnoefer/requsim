from . import WorldObject


class Qubit(WorldObject):
    """A Qubit.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    station : Station
        The station at which the qubit is located.
    unresolved_noise : NoiseChannel or None
        Noise that affected the qubit, but has not been applied to the state
        yet. None means nothing is unresolved. Default: None
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    pair : Pair or None
        Pair if the qubit is part of a Pair, None else.
    station : Station
        The station at which the qubit is located.
    type : str
        "Qubit"
    unresolved_noise : NoiseChannel or None

    """

    # station should also know about which qubits are at its location
    def __init__(self, world, station, unresolved_noise=None, label=None):
        self.station = station
        self.unresolved_noise = unresolved_noise
        super(Qubit, self).__init__(world=world, label=label)
        self.pair = None

    def __str__(self):
        return f"{self.label} at station {self.station.label if self.station else self.station}, part of pair {self.pair.label if self.pair else self.pair}."

    @property
    def type(self):
        return "Qubit"

    def destroy(self):
        # station needs to be notified that qubit is no longer there, not sure how to handle pairs yet
        self.station.remove_qubit(self)
        super(Qubit, self).destroy()
