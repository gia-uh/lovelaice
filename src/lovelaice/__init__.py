"""Lovelaice — a sovereign coding agent for the terminal."""
from .config import Config
from .core import Lovelaice
from .workflows import workflow

__version__ = "1.0.0"
__all__ = ["Config", "Lovelaice", "workflow", "__version__"]
