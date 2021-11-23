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
    memory_noise : NoiseChannel or None
    memory_cutoff_time : scalar or None
    resource_tracking : defaultdict
        Intermediate store for carrying over resources used by discarded
        pairs/qubits.
    BSM_noise_model : NoiseModel
    creation_noise_channel : NoiseChannel or None
    dark_count_probability : scalar

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
        label=None,
    ):
        self.position = position
        self.qubits = []
        self.memory_noise = memory_noise
        self.memory_cutoff_time = memory_cutoff_time
        self.BSM_noise_model = BSM_noise_model
        self.creation_noise_channel = creation_noise_channel
        self.dark_count_probability = dark_count_probability
        super(Station, self).__init__(world=world, label=label)

    def __repr__(self):
        return self.__class__.__name__ + (
            f"(world={self.world}, position={self.position}, "
            f"memory_noise={self.memory_noise}, "
            f"memory_cutoff_time={self.memory_cutoff_time}, "
            f"BSM_noise_model={self.BSM_noise_model}, "
            f"creation_noise_channel={self.creation_noise_channel}, "
            f"dark_count_probability={self.dark_count_probability}, "
            f"label={self.label})"
        )

    @property
    def type(self):
        return "Station"

    def register_qubit(self, qubit):
        self.qubits += [qubit]
        qubit.update_info({"station": self})
        qubit.add_destroy_callback(self.remove_qubit)
        if self.memory_noise is not None:
            qubit.add_time_dependent_noise(self.memory_noise)

    def create_qubit(self, label=None):
        """Create a new qubit at this station.

        Returns
        -------
        Qubit
            The created Qubit object.

        """
        if self.creation_noise_channel is not None:
            new_qubit = Qubit(
                world=self.world,
                unresolved_noises=[self.creation_noise_channel],
                label=label,
            )
        else:
            new_qubit = Qubit(world=self.world, label=label)
        self.register_qubit(new_qubit)
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
