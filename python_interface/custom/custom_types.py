"""Custom types for better type hints"""

from typing import TypeVar, Final, LiteralString
from pathlib import Path
from os import PathLike as _PathLike

PathLike: Final[TypeVar] = TypeVar("PathLike", str, LiteralString, Path, _PathLike)
"""Path-like objects"""
