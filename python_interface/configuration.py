"""General configuration of the framework."""
import constants
import os
import logging

from custom_types import PathLike, Backend, Topology
from json_generator import FFjson
from typing import List, Dict, Union, Optional, Any, get_args
from custom_exceptions import MutuallyExclusiveArgumentsException

# TODO: make paths more flexible
FFL_DIR = "/mnt/shared/gmittone/FastFederatedLearning/"
EXECUTABLE_PATH_MS = FFL_DIR + "build/examples/masterworker/masterworker_dist"
EXECUTABLE_PATH_P2P = FFL_DIR + "build/examples/p2p/p2p_dist"


class Configuration(dict):

    def __init__(self, json_path: PathLike, data_path: PathLike, runner_path: PathLike,
                 torchscript_path: PathLike, topology: Topology = constants.MASTER_WORKER,
                 endpoints: Union[int, List[str], List[Dict[str, str]]] = 2,
                 commands: Optional[Union[str, List[str]]] = None, backend: Backend = constants.TCP,
                 force_cpu: bool = True, rounds: int = 1, epochs: int = 1):
        super().__init__()

        # TODO: Add logging

        self.json: FFjson = FFjson(topology=topology, endpoints=endpoints, commands=commands)

        self.set_json_path(json_path=json_path)
        self.set_data_path(data_path=data_path)
        self.set_runner_path(runner_path=runner_path)
        self.set_executable_path(topology=topology)
        self.set_torchscript_path(torchscript_path=torchscript_path)
        self.set_backend(backend=backend)
        self.set_force_cpu(force_cpu=force_cpu)
        self.set_rounds(rounds=rounds)
        self.set_epochs(epochs=epochs)

    # Getters
    def get_json_path(self) -> PathLike:
        return self["json_path"]

    def get_data_path(self) -> PathLike:
        return self["data_path"]

    def get_runner_path(self) -> PathLike:
        return self["runner_path"]

    def get_executable_path(self) -> PathLike:
        return self["executable_path"]

    def get_torchscript_path(self) -> PathLike:
        return self["torchscript_path"]

    def get_backend(self) -> str:
        return self["backend"]

    def get_force_cpu(self) -> bool:
        return True if self["force_cpu"] else 0

    def get_rounds(self) -> int:
        return self["rounds"]

    def get_epochs(self) -> int:
        return self["epochs"]

    def get_json(self) -> FFjson:
        return self.json

    # Setters
    def set_json_path(self, json_path: PathLike):
        check_and_create_path(json_path, "json_path")
        self["json_path"]: PathLike = json_path

    def set_data_path(self, data_path: PathLike):
        check_and_create_path(data_path, "data_path")
        self["data_path"]: PathLike = data_path

    def set_runner_path(self, runner_path: PathLike):
        check_and_create_path(runner_path, "runner_path")
        self["runner_path"]: PathLike = runner_path

    def set_executable_path(self, executable_path: Optional[PathLike] = None, topology: Optional[Topology] = None):
        check_mutually_exclusive_args(executable_path, topology)
        if executable_path is not None:
            check_and_create_path(executable_path, "executable_path")
            self["executable_path"]: PathLike = executable_path
        else:  # TODO: Make this choice extendible in the future with user-defined topologies
            check_var_in_literal(topology, Topology)
            self["executable_path"]: PathLike = \
                EXECUTABLE_PATH_MS if topology == constants.MASTER_WORKER else EXECUTABLE_PATH_P2P

    def set_torchscript_path(self, torchscript_path: PathLike):
        check_and_create_path(torchscript_path, "torchscript_path")
        self["torchscript_path"]: PathLike = torchscript_path

    def set_backend(self, backend: Backend):
        check_var_in_literal(backend, Backend)
        self["backend"]: Backend = backend

    def set_force_cpu(self, force_cpu: bool):
        self["force_cpu"]: bool = 1 if force_cpu else 0

    def set_rounds(self, rounds: int):
        check_positive_int(rounds)
        self["rounds"]: int = rounds

    def set_epochs(self, epochs: int):
        check_positive_int(epochs)
        self["epochs"]: int = epochs


# Utility
def check_and_create_path(path: PathLike, target: str = ""):
    dirname = os.path.dirname(path)
    logging.debug("Attempting to create " + target + " at path: " + str(path) + "...")
    if os.path.exists(dirname):
        logging.debug("Path: " + str(path) + " already existing.")
    else:
        logging.debug("Path: " + str(path) + " not found...")
        os.makedirs(dirname, exist_ok=True)
        logging.debug("Created path: " + str(path) + ".")


def check_mutually_exclusive_args(arg_1: Any, arg_2: Any):
    if (arg_1 is not None and arg_2 is not None) or (arg_1 is None and arg_2 is None):
        raise MutuallyExclusiveArgumentsException("Mutually exclusive arguments.")


def check_var_in_literal(var: Any, literal: Any):
    if var not in get_args(literal):
        raise ValueError("Value " + str(var) + " not in " + str(literal))


def check_positive_int(var: int, threshold: int = 0):
    if var <= threshold:
        raise ValueError("Value " + str(var) + " must be greater than" + str(threshold))
