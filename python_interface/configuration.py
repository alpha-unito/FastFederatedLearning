"""General configuration of the FastFederatedLearning runtime."""
import constants
import logging
import utils

from python_interface.custom.custom_types import PathLike, Backend, Topology
from json_generator import JSONGenerator
from typing import List, Dict, Union, Optional

# TODO: make paths more flexible
FFL_DIR = "/mnt/shared/gmittone/FastFederatedLearning/"
"""FastFederatedLearning root directory"""
EXECUTABLE_PATH_MS = FFL_DIR + "build/examples/masterworker/masterworker_dist"
"""Master-Worker executable path"""
EXECUTABLE_PATH_P2P = FFL_DIR + "build/examples/p2p/p2p_dist"
"""Peer-to-Peer executable path"""


class Configuration(dict):
    """General configuration of the FastFederatedLearning runtime."""

    def __init__(self, json_path: PathLike, data_path: PathLike, runner_path: PathLike,
                 torchscript_path: PathLike, topology: Topology = constants.MASTER_WORKER,
                 endpoints: Union[int, List[str], List[Dict[str, str]]] = 2,
                 commands: Optional[Union[str, List[str]]] = None, backend: Backend = constants.TCP,
                 force_cpu: bool = True, rounds: int = 1, epochs: int = 1):
        """General configuration of FastFederatedLearning.
        This class directly extends a standard Python dictionary.

        :param json_path: path of the JSON configuration file.
        :type json_path: PathLike
        :param data_path: path of the dataset.
        :type data_path: PathLike
        :param runner_path: path of the FastFlow executable file (masterworker or p2p).
        :type runner_path: PathLike
        :param torchscript_path: path of the TorchScript model.
        :type torchscript_path: PathLike
        :param topology: type of topology for the experiment (masterworker or p2p).
        :type topology: Topology
        :param endpoints: number or list of hosts to add to the federation.
        :type endpoints: Union[int, List[str], List[Dict[str, str]]]
        :param commands: pre-command (or list of pre-commands) to add to the JSON configuration file.
        :type commands: Optional[Union[str, List[str]]]
        :param backend: type of communication backend.
        :type backend: Backend
        :param force_cpu: force the workload on the CPU.
        :type force_cpu: bool
        :param rounds: number of federated rounds to be run.
        :type rounds: int
        :param epochs: number of epochs to be run.
        :type epochs: int
        """
        super().__init__()
        self.logger: logging.Logger = utils.get_logger(self.__class__.__name__)

        self.logger.info("Generating JSON configuration file...")
        self.json: JSONGenerator = JSONGenerator(topology=topology, endpoints=endpoints, commands=commands)

        self.set_json_path(json_path=json_path)
        self.set_data_path(data_path=data_path)
        self.set_runner_path(runner_path=runner_path)
        self.set_executable_path(topology=topology)
        self.set_torchscript_path(torchscript_path=torchscript_path)
        self.set_backend(backend=backend)
        self.set_force_cpu(force_cpu=force_cpu)
        self.set_rounds(rounds=rounds)
        self.set_epochs(epochs=epochs)

    def get_json_path(self) -> PathLike:
        """Get the JSON configuration file path.

        :return: the JSON configuration file path
        :rtype: PathLike
        """
        return self["json_path"]

    def get_data_path(self) -> PathLike:
        """Get the dataset path.

        :return: the dataset path
        :rtype: PathLike
        """
        return self["data_path"]

    def get_runner_path(self) -> PathLike:
        """Get the DFF_run executable path.

        :return: the DFF_run executable path
        :rtype: PathLike
        """
        return self["runner_path"]

    def get_executable_path(self) -> PathLike:
        """Get the chosen FastFlow executable path.

        :return: the chosen FastFlow executable path
        :rtype: PathLike
        """
        return self["executable_path"]

    def get_torchscript_path(self) -> PathLike:
        """Get the TorchScript model path.

        :return: the TorchScript model path
        :rtype: PathLike
        """
        return self["torchscript_path"]

    def get_backend(self) -> str:
        """Get the chosen backend.

        :return: the backend name.
        :rtype: str
        """
        return self["backend"]

    def get_force_cpu(self) -> bool:
        """Get if the workload is to be forced on the CPU.

        :return: True if the workload is to be forced on the CPU.
        :rtype: bool
        """
        return True if self["force_cpu"] else 0

    def get_rounds(self) -> int:
        """Get the number of federated rounds to be run.

        :return: the number of federated rounds to be run.
        :rtype: int
        """
        return self["rounds"]

    def get_epochs(self) -> int:
        """Get the number of epochs to be run.

        :return: the number of epochs to be run.
        :rtype: int
        """
        return self["epochs"]

    def get_json(self) -> JSONGenerator:
        """Get the JSON configuration object.

        :return: JSON configuration object.
        :rtype: JSONGenerator
        """
        return self.json

    def set_json_path(self, json_path: PathLike):
        """Set the JSON configuration file path.

        :param json_path: JSON configuration file path.
        :type json_path: PathLike
        """
        utils.check_and_create_path(json_path, "json_path", self.logger)
        self.logger.info("Setting the JSON config file path to %s", json_path)
        self["json_path"]: PathLike = json_path

    def set_data_path(self, data_path: PathLike):
        """Set the dataset path.

        :param data_path: dataset path.
        :type data_path: PathLike
        """
        utils.check_and_create_path(data_path, "data_path", self.logger)
        self.logger.info("Setting the dataset path to %s", data_path)
        self["data_path"]: PathLike = data_path

    def set_runner_path(self, runner_path: PathLike):
        """Set the DFF_run path.

        :param runner_path: DFF_run path.
        :type runner_path: PathLike
        """
        utils.check_and_create_path(runner_path, "runner_path", self.logger)
        self.logger.info("Setting the DFF_run path to %s", runner_path)
        self["runner_path"]: PathLike = runner_path

    def set_executable_path(self, executable_path: Optional[PathLike] = None, topology: Optional[Topology] = None):
        """Set the FastFlow executable path.

        :param executable_path: FastFlow executable path.
        :type executable_path: Optional[PathLike]
        :param topology: type of topology chosen.
        :type topology: Optional[Topology]
        """
        utils.check_mutually_exclusive_args(executable_path, topology, self.logger)
        if executable_path is not None:
            utils.check_and_create_path(executable_path, "executable_path", self.logger)
            self.logger.info("Setting the FastFlow executable path to %s", executable_path)
            self["executable_path"]: PathLike = executable_path
        else:  # TODO: Make this choice extendible in the future with user-defined topologies
            utils.check_var_in_literal(topology, Topology, self.logger)
            self.logger.info("Setting the FastFlow executable path to %s", executable_path)
            self["executable_path"]: PathLike = \
                EXECUTABLE_PATH_MS if topology == constants.MASTER_WORKER else EXECUTABLE_PATH_P2P

    def set_torchscript_path(self, torchscript_path: PathLike):
        """Set the TorchScript model path.

        :param torchscript_path: TorchScript model path.
        :type torchscript_path: PathLike
        """
        utils.check_and_create_path(torchscript_path, "torchscript_path", self.logger)
        self.logger.info("Setting the TorchScript model path to %s", torchscript_path)
        self["torchscript_path"]: PathLike = torchscript_path

    def set_backend(self, backend: Backend):
        """Set the communication backend.

        :param backend: communication backend.
        :type backend: Backend
        """
        utils.check_var_in_literal(backend, Backend, self.logger)
        self.logger.info("Setting the communication backend to %s", backend)
        self["backend"]: Backend = backend

    def set_force_cpu(self, force_cpu: bool):
        """Set to the workload has to be offloaded entirely to the CPU.

        :param force_cpu: offload the workload on the CPU.
        :type force_cpu: bool
        """
        self.logger.info("Forcing the workload on the CPU: %s", force_cpu)
        self["force_cpu"]: bool = 1 if force_cpu else 0

    def set_rounds(self, rounds: int):
        """Set number of federated rounds to be run.

        :param rounds: number of federated rounds to be run.
        :type rounds: int
        """
        utils.check_positive_int(rounds, 0, self.logger)
        self.logger.info("Setting the number of federated rounds to %s", rounds)
        self["rounds"]: int = rounds

    def set_epochs(self, epochs: int):
        """Set number of epochs to be run.

        :param epochs: number of epochs to be run.
        :type epochs: int
        """
        utils.check_positive_int(epochs, 0, self.logger)
        self.logger.info("Setting the number of epochs to %s", epochs)
        self["epochs"]: int = epochs
