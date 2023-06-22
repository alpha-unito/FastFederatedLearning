"""General configuration of the framework."""
import constants
from custom_types import PathLike
from json_generator import FFjson
from typing import List, Dict, Union, Optional

EXECUTABLE_PATH_MS = "/mnt/shared/gmittone/FastFederatedLearning/build/examples/masterworker/masterworker_dist"
EXECUTABLE_PATH_P2P = "/mnt/shared/gmittone/FastFederatedLearning/build/examples/p2p/p2p_dist"


class Configuration(dict):

    def __init__(self, json_path: PathLike, data_path: PathLike, runner_path: PathLike,
                 topology: str = constants.MASTER_WORKER, endpoints: Union[int, List[str], List[Dict[str, str]]] = 2,
                 commands: Optional[Union[str, List[str]]] = None, torchscript_path: PathLike= None):
        super().__init__()

        self.json: FFjson = FFjson(topology=topology, endpoints=endpoints, commands=commands)

        self["json_path"]: PathLike = json_path
        self["data_path"]: PathLike = data_path
        self["runner_path"]: PathLike = runner_path
        self[
            "executable_path"]: PathLike = EXECUTABLE_PATH_MS if self.json.topology == constants.MASTER_WORKER else EXECUTABLE_PATH_P2P

    def get_json_path(self) -> PathLike:
        return self["json_path"]

    def get_data_path(self) -> PathLike:
        return self["data_path"]

    def get_runner_path(self) -> PathLike:
        return self["runner_path"]

    def get_executable_path(self) -> PathLike:
        return self["executable_path"]

    def get_json(self) -> FFjson:
        return self.json

    def set_json_path(self, json_path: PathLike):
        self["json_path"] = json_path

    def set_data_path(self, data_path: PathLike):
        self["data_path"] = data_path

    def set_runner_path(self, runner_path: PathLike):
        self["runner_path"] = runner_path

    def set_executable_path(self, executable_path: PathLike):
        self["executable_path"] = executable_path
