"""Constants useful for not using string all over the place"""

from typing import Final, List, Callable

# Topology-related constants #
MASTER_WORKER: Final[str] = "masterworker"
"""Centralized (master-worker) federation topology"""

PEER_TO_PEER: Final[str] = "peer_to_peer"
"""Peer to peer federation topology"""

# JSON-related constants #
DEFAULT_ENDPOINT: Final[str] = "localhost"
"""Endpoint placeholder"""

PRE_CMD: Final[str] = "preCmd"
"""preCmd field name of the FastFlow json format"""

ENDPOINT: Final[str] = "endpoint"
"""endpoint field name of the FastFlow json format"""

NAME: Final[str] = "name"
"""name field name of the FastFlow json format"""

GROUPS: Final[str] = "groups"
"""group field name of the FastFlow json format"""

FEDERATOR: Final[str] = "Federator"
"""Server name for the FastFlow json format"""

WORKER: Callable[[int], str] = lambda rank: "W" + str(rank)
"""Function for determining the workers names in the FastFlow json format"""

# Backend constants #
TCP: Final[str] = "TCP"
"""TCP communication backend"""

MPI: Final[str] = "MPI"
"""MPI communication backend"""
