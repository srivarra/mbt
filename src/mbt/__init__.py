from importlib.metadata import version

from ._core import hello_world

__all__ = ["hello_world"]

__version__ = version("mbt")


def main() -> None:
    print(hello_world())
