from importlib.metadata import version

from . import im
from .core.mibifile import MibiFile

__all__ = ["MibiFile", "im"]

__version__ = version("mbt")
