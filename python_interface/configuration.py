"""General configuration of the framework."""
import constants
from custom_types import PathLike
from json_generator import FFjson
from typing import List, Dict, Union, Optional

from constants import TCP

FFL_DIR = "/mnt/shared/gmittone/FastFederatedLearning/"
EXECUTABLE_PATH_MS = FFL_DIR + "build/examples/masterworker/masterworker_dist"
EXECUTABLE_PATH_P2P = FFL_DIR + "build/examples/p2p/p2p_dist"


class Configuration(dict):

    def __init__(self, json_path: PathLike, data_path: PathLike, runner_path: PathLike, config_file_path: PathLike,
                 torchscript_path: PathLike, topology: str = constants.MASTER_WORKER,
                 endpoints: Union[int, List[str], List[Dict[str, str]]] = 2,
                 commands: Optional[Union[str, List[str]]] = None, backend: str = TCP, force_cpu: bool = True,
                 rounds: int = 1, epochs: int = 1):
        super().__init__()

        self.json: FFjson = FFjson(topology=topology, endpoints=endpoints, commands=commands)

        self["json_path"]: PathLike = json_path
        self["data_path"]: PathLike = data_path
        self["runner_path"]: PathLike = runner_path
        self["config_file_path"]: PathLike = config_file_path
        self[
            "executable_path"]: PathLike = EXECUTABLE_PATH_MS if self.json.topology == constants.MASTER_WORKER else EXECUTABLE_PATH_P2P
        self["torchscript_path"]: PathLike = torchscript_path
        self["backend"]: str = backend  # TODO: controllare che backend sia TCP o MPI
        self["force_cpu"]: int = 1 if force_cpu else 0
        self["rounds"]: int = rounds  # TODO check >=1
        self["epochs"]: int = epochs  # TODO check >=1

    def get_json_path(self) -> PathLike:
        return self["json_path"]

    def get_data_path(self) -> PathLike:
        return self["data_path"]

    def get_runner_path(self) -> PathLike:
        return self["runner_path"]

    def get_config_file_path(self) -> PathLike:
        return self["config_file_path"]

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

    def set_json_path(self, json_path: PathLike):
        self["json_path"] = json_path

    def set_data_path(self, data_path: PathLike):
        self["data_path"] = data_path

    def set_runner_path(self, runner_path: PathLike):
        self["runner_path"] = runner_path

    def set_config_file_path(self, config_file_path: PathLike):
        self["config_file_path"] = config_file_path

    def set_executable_path(self, executable_path: PathLike):
        self["executable_path"] = executable_path

    def set_torchscript_path(self, torchscript_path: PathLike):
        self["torchscript_path"] = torchscript_path

    def set_backend(self, backend: str):
        self["backend"] = backend

    def set_force_cpu(self, force_cpu: bool):
        self["force_cpu"] = force_cpu

    def set_rounds(self, rounds: int):
        self["rounds"] = rounds

    def set_epochs(self, epochs: int):
        self["epochs"] = epochs
