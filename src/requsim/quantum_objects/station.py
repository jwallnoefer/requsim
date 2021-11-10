from . import WorldObject, Qubit
from ..noise import NoiseModel
from collections import defaultdict
from warnings import warn
from .. import events


class Station(WorldObject):
    """A repeater station.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    memory_noise : callable or None
        Should take parameters rho (density matrix) and t (time). Default: None
    memory_cutoff_time : scalar or None
        Qubits will be discarded after this amount of time in memory.
        Default: None
    BSM_noise_model : NoiseModel
        Noise model that is used for Bell State measurements performed at this
        station (especially for entanglement swapping).
        Default: dummy NoiseModel that corresponds to no noise.
    creation_noise_channel : NoiseChannel or None
        Noise channel that is applied to a qubit on creation. (e.g. misalignment)
        Default: None
    dark_count_probability : scalar
        Probability that a detector clicks without a state arriving.
        This is not used by the Station itself, but state generation functions
        may use this. Default: 0
    id : int or None
        [DEPRECATED: Do not use!] Label for the station. Default: None
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    qubits : list of Qubit objects
        The qubits currently at this position.
    type : str
        "Station"
    memory_noise : callable or None
    memory_cutoff_time : callable or None
    resource_tracking : defaultdict
        Intermediate store for carrying over resources used by discarded
        pairs/qubits.
    BSM_noise_model : NoiseModel
    creation_noise_channel : NoiseChannel or None
    dark_count_probability : scalar
    id : int or None

    """

    def __init__(
        self,
        world,
        position,
        memory_noise=None,
        memory_cutoff_time=None,
        BSM_noise_model=NoiseModel(),
        creation_noise_channel=None,
        dark_count_probability=0,
        id=None,
        label=None,
    ):
        self.id = id
        self.position = position
        self.qubits = []
        self.resource_tracking = defaultdict(
            lambda: {"resource_cost_add": 0, "resource_cost_max": 0}
        )
        self.memory_noise = memory_noise
        self.memory_cutoff_time = memory_cutoff_time
        self.BSM_noise_model = BSM_noise_model
        self.creation_noise_channel = creation_noise_channel
        self.dark_count_probability = dark_count_probability
        super(Station, self).__init__(world=world, label=label)

    def __str__(self):
        return f"{self.label} at position {self.position}."

    @property
    def type(self):
        return "Station"

    def create_qubit(self):
        """Create a new qubit at this station.

        Returns
        -------
        Qubit
            The created Qubit object.

        """
        new_qubit = Qubit(
            world=self.world, station=self, unresolved_noise=self.creation_noise_channel
        )
        self.qubits += [new_qubit]
        if self.memory_cutoff_time is not None:
            discard_event = events.DiscardQubitEvent(
                time=self.event_queue.current_time + self.memory_cutoff_time,
                qubit=new_qubit,
            )
            self.event_queue.add_event(discard_event)
        return new_qubit

    def remove_qubit(self, qubit):
        try:
            self.qubits.remove(qubit)
        except ValueError:
            warn(
                "Tried to remove qubit {} from station {}, but the station was not tracking that qubit.".format(
                    repr(qubit), repr(self)
                )
            )
            # print(self.event_queue.current_time)
            # print(self.event_queue.queue)
            # print(self.world.world_objects)
