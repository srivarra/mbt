from importlib.metadata import version

from . import im
from ._core import hello_world
from .core.mibifile import MibiFile

__all__ = ["hello_world", "MibiFile", "im"]

__version__ = version("mbt")
