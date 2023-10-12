"""Constants useful for not using string all over the place"""

from typing import Final, Callable, LiteralString, Tuple, Literal

from python_interface.custom.custom_types import PathLike

# Version #
VERSION: Final[LiteralString] = "v0.1.0-alpha"
"""FastFL software version"""

# Paths #
# TODO: this is not general
DEFAULT_FFL_DIR: Final[PathLike] = "/mnt/shared/gmittone/FastFederatedLearning/"
"""FastFederatedLearning deault  root directory"""

DEFAULT_WORKSPACE_DIR: Final[PathLike] = DEFAULT_FFL_DIR + "workspace/"
"""FastFederatedLearning default workspace directory"""

DEFAULT_LIBS_DIR: Final[PathLike] = DEFAULT_FFL_DIR + "libs/"
"""FastFederatedLearning default libraries directory"""

DEFAULT_BUILD_DIR: Final[PathLike] = DEFAULT_FFL_DIR + "build/"
"""FastFederatedLearning default libraries directory"""

DEFAULT_JSON_PATH: Final[PathLike] = DEFAULT_WORKSPACE_DIR + "config.json"
"""FastFederatedLearning default JSON configuration file path"""

DEFAULT_MODEL_PATH: Final[PathLike] = DEFAULT_WORKSPACE_DIR + "model.pt"
"""FastFederatedLearning default model file path"""

DEFAULT_DFF_RUN_PATH: Final[PathLike] = DEFAULT_LIBS_DIR + "fastflow/ff/distributed/loader/dff_run"
"""FastFederatedLearning default dff_run file path"""

EXECUTABLE_PATH_MS: Final[PathLike] = DEFAULT_BUILD_DIR + "C/examples/masterworker/masterworker_dist"
"""Master-Worker executable path"""

EXECUTABLE_PATH_P2P: Final[PathLike] = DEFAULT_BUILD_DIR + "C/examples/p2p/p2p_dist"
"""Peer-to-Peer executable path"""

EXECUTABLE_PATH_EI: Final[PathLike] = DEFAULT_BUILD_DIR + "C/examples/edgeinference/edgeinference_dist"
"""Basic edge-inference executable path"""

EXECUTABLE_PATH_MVDET: Final[PathLike] = DEFAULT_BUILD_DIR + "C/examples/mvdet/mvdet_dist"
"""MvDet edge-inference executable path"""

# Logging #
LOGGING_CONFIGURATION: Final[PathLike] = DEFAULT_FFL_DIR + "python_interface/logging.conf"
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

WORKER: Callable[[int], LiteralString] = lambda rank: "W" + str(rank)
"""Function for determining the workers names in the FastFlow json format"""

SOURCE: Callable[[int], LiteralString] = lambda rank: "S" + str(rank)
"""Function for determining the source names in the FastFlow json format (mvdet)"""

AGGREGATOR: Callable[[int], LiteralString] = lambda rank: "A" + str(rank)
"""Function for determining the aggregator names in the FastFlow json format (mvdet)"""

# Backend constants #
TCP: Final[LiteralString] = "TCP"
"""TCP communication backend"""

MPI: Final[LiteralString] = "MPI"
"""MPI communication backend"""

Backend: Literal[Final[Tuple[LiteralString, LiteralString]]] = Literal[TCP, MPI]
"""Supported backend names"""
