from importlib.metadata import version

from ._core import hello_world

__all__ = ["hello_world"]

__version__ = version("mbt")
