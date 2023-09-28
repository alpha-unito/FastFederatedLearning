"""Constants useful for not using string all over the place"""

from typing import Final, Callable, LiteralString, Tuple, Literal
from python_interface.custom.custom_types import PathLike

# Version #
VERSION: Final[LiteralString] = "v0.1.0-alpha"
"""FastFL software version"""

# Paths #
FFL_DIR: Final[PathLike] = "/mnt/shared/gmittone/FastFederatedLearning/"
"""FastFederatedLearning root directory"""

EXECUTABLE_PATH_MS: Final[PathLike] = FFL_DIR + "build/C/examples/masterworker/masterworker_dist"
"""Master-Worker executable path"""

EXECUTABLE_PATH_P2P: Final[PathLike] = FFL_DIR + "build/C/examples/p2p/p2p_dist"
"""Peer-to-Peer executable path"""

EXECUTABLE_PATH_EI: Final[PathLike] = FFL_DIR + "build/C/examples/edgeinference/edgeinference"
"""Basic edge-inference executable path"""

EXECUTABLE_PATH_MVDET: Final[PathLike] = FFL_DIR + "build/C/examples/mvdet/mvdet"
"""MvDet edge-inference executable path"""

# Logging #
LOGGING_CONFIGURATION: Final[PathLike] = "python_interface/logging.conf"
"""Path to the logging configuration file"""

# Topology-related constants #
MASTER_WORKER: Final[LiteralString] = "masterworker"
"""Centralized (master-worker) federation topology"""

PEER_TO_PEER: Final[LiteralString] = "peer_to_peer"
"""Peer to peer federation topology"""

EDGE_INFERENCE: Final[LiteralString] = "edge_inference"
"""Basic edge inference topology"""

MVDET: Final[LiteralString] = "multiview_detection"
"""Multiview detection topology"""

Topology: Literal[Final[Tuple[LiteralString]]] = Literal[MASTER_WORKER, PEER_TO_PEER, EDGE_INFERENCE, MVDET]
"""Supported topologies names"""

# JSON-related constants #
DEFAULT_ENDPOINT: Final[LiteralString] = "localhost"
"""Endpoint placeholder"""

PRE_CMD: Final[LiteralString] = "preCmd"
"""preCmd field name of the FastFlow json format"""

ENDPOINT: Final[LiteralString] = "endpoint"
"""endpoint field name of the FastFlow json format"""

NAME: Final[LiteralString] = "name"
"""name field name of the FastFlow json format"""

GROUPS: Final[LiteralString] = "groups"
"""group field name of the FastFlow json format"""

FEDERATOR: Final[LiteralString] = "Federator"
"""Server name for the FastFlow json format"""

WORKER: Callable[[int], LiteralString] = lambda rank: "W" + str(rank)
"""Function for determining the workers names in the FastFlow json format"""

# Backend constants #
TCP: Final[LiteralString] = "TCP"
"""TCP communication backend"""

MPI: Final[LiteralString] = "MPI"
"""MPI communication backend"""

Backend: Literal[Final[Tuple[LiteralString, LiteralString]]] = Literal[TCP, MPI]
"""Supported backend names"""
