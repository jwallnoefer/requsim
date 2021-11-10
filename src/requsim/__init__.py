try:
    from . import version

    __version__ = version.version
except ImportError:
    __version__ = "unknown"

from . import events, noise, quantum_objects, world
