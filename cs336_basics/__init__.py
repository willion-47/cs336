import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    # The assignment is commonly run directly from a source checkout, where
    # distribution metadata does not exist yet.
    __version__ = "0.1.0"
