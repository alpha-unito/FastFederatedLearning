"""Custom types for better type hints"""

import constants

from typing import TypeVar, Final, Literal
from pathlib import Path
from os import PathLike as _PathLike

PathLike: Final[TypeVar] = TypeVar("PathLike", str, Path, _PathLike)
"""Path-like objects"""

Backend = Literal[constants.TCP, constants.MPI]
"""Supported backend names"""

Topology = Literal[constants.MASTER_WORKER, constants.PEER_TO_PEER]
"""Supported topologies names"""
