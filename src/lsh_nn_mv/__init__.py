"""Top-level package for LSH-NN-MV."""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - metadata query
    __version__ = version("lsh-nn-mv")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
