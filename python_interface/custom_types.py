"""Custom types for better type hints"""

from typing import TypeVar
from pathlib import Path
from typing import Final

PathLike: Final[TypeVar] = TypeVar("PathLike", str, Path)
"""Path-like objects"""
