from . import WorldObject
from collections import defaultdict


class Qubit(WorldObject):
    """A Qubit.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    unresolved_noises : list of NoiseChannel
        Noise that affected the qubit, but has not been applied to the state
        yet. Default: []
    info : dict
        Initial information dictionary for storing additional information such
        as where the qubit is located and whether it is part of a pair. This
        should not be used as a workaround to access these actions
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    type : str
        "Qubit"

    """

    # station should also know about which qubits are at its location
    def __init__(self, world, unresolved_noises=None, info={}, label=None):
        self._unresolved_noises = []
        self._info = defaultdict(lambda: None)
        self._info.update(info)
        self._noise_handlers = []
        self._time_dependent_noises = []
        super(Qubit, self).__init__(world=world, label=label)

    def __repr__(self):
        return self.__class__.__name__ + (
            f"(world={self.world}, unresolved_noise={self._unresolved_noises}, "
            f"info={self._info}, label={self.label})"
        )

    def __str__(self):
        return f"{self.label} at station {self.station.label if self.station else self.station}, part of pair {self.pair.label if self.pair else self.pair}."

    @property
    def type(self):
        return "Qubit"

    def update_info(self, info_dictionary):
        """Update to the internal info dict.

        This method is provided to avoid accessing `self._info` directly,
        as the information in `self._info` should not be used to avoid using
        the proper registering and deregistering methods to e.g. handle noise.

        Parameters
        ----------
        info_dictionary : dict
            Internal info dict will be updated with this dictionary.

        Returns
        -------
        None

        """
        self._info.update(info_dictionary)

    def add_time_dependent_noise(self, noise_channel):
        self._time_dependent_noises += [noise_channel]

    def remove_time_dependent_noise(self, noise_channel):
        self._time_dependent_noises.remove(noise_channel)

    def add_noise_handler(self, noise_handler):
        # try handling unresolved noise with the new noise_handler
        for unresolved_noise in list(self._unresolved_noises):
            handling_successful = noise_handler(unresolved_noise)
            if handling_successful:
                self._unresolved_noises.remove(unresolved_noise)
        self._noise_handlers += [noise_handler]

    def remove_noise_handler(self, noise_handler):
        try:
            self._noise_handlers.remove(noise_handler)
        except ValueError:  # happens if already removed
            pass

    def _delegate_noise_handling(self, noise_channel, *args, **kwargs):
        handling_successful = False
        for noise_handler in self._noise_handlers:
            # try different noisehandlers until one works
            handling_successful = noise_handler(noise_channel, *args, **kwargs)
            if handling_successful:
                break
        return handling_successful

    def _on_update_time(self):
        time_interval = self.event_queue.current_time - self.last_updated
        for time_dependent_noise in self._time_dependent_noises:
            handling_successful = self._delegate_noise_handling(
                time_dependent_noise, t=time_interval
            )
