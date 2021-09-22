try:
    from . import version

    __version__ = version.version
except ImportError:
    __version__ = "unknown"
