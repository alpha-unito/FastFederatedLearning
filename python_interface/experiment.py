"""
Class responsible for organizing, running, and eventually shutting down, the federation.
"""
import logging
from subprocess import call, SubprocessError
from typing import List, NoReturn, Optional

import torch
from pssh.clients.ssh.parallel import ParallelSSHClient
from torch.jit import ScriptModule

from python_interface.configuration import Configuration
from python_interface.custom.custom_exceptions import WronglySpecifiedArgumentException
from python_interface.dataset import Dataset
from python_interface.json_generator import JSONGenerator
from python_interface.model import Model
from python_interface.utils import utils, constants


class Experiment:
    """Object handling the experiment"""

    def __init__(self, configuration: Configuration, dataset: Dataset, model: Optional[Model] = None):
        """Object serving as interface for handling the Federated Learning experiment.

        :param configuration: a federation configuration file.
        :type configuration: Configuration
        :param model: a compiled PyTorch Model.
        :type model: ScriptModule
        """
        self.logger: logging.Logger = utils.get_logger(self.__class__.__name__)

        self.configuration: Configuration = configuration
        self.json: JSONGenerator = self.configuration.get_json()
        self.model: Optional[Model] = model if model is not None else Model()
        self.dataset: Dataset = dataset

        self.logger.info("Experiment set up correctly.")

    def run_experiment(self) -> NoReturn | WronglySpecifiedArgumentException | SubprocessError:
        """Experiment start-up.
        Saves the TorchScript model and the JSON configuration file, if it not already saved, and then calls the C/C++ backend executable.
        """
        try:
            self.model.check_torchscript_no_model(self.configuration.get_torchscript_path())
        except WronglySpecifiedArgumentException as e:
            self.logger.critical("The specified model configuration is not correct: %s", e)
            raise e
        if not self.model.already_exists():
            self.logger.info("Saving TorchScript model to: %s", self.configuration.get_torchscript_path())
            torch.jit.save(self.model.compile(), self.configuration.get_torchscript_path())

        self.json.generate_json_file(self.configuration.get_json_path())

        dff_run_command: List[str] = self.create_dff_run_command()
        self.logger.info("Launching the FastFlow backend: %s", dff_run_command)
        self.logger.info('-' * 80)
        try:
            call(dff_run_command)
        except SubprocessError as e:
            self.logger.critical("Call to FastFlow backend failed: %s", e)
            self.logger.info('-' * 80)
            self.logger.info("Experiment not completed correctly.")
            raise e
        self.logger.info('-' * 80)
        self.logger.info("Experiment completed correctly.")

    def create_dff_run_command(self) -> List[str]:
        """Method to create the command-line string for executing the experiment through DFF_run.

        :param topology: the chosen experiment topology.
        :type topology: Topology
        :return: list of the exit codes received by the contacted hosts.
        :rtype: List[int]
        """
        self.logger.info("Creating the DFF_run command...")
        dff_run_command: List[str] = [self.configuration.get_runner_path(),
                                      "-V",
                                      "-p",
                                      self.configuration.get_backend(),
                                      "-f ",
                                      self.configuration.get_json_path(),
                                      self.configuration.get_executable_path()]

        self.logger.info("Adding the %s command line parameters...", self.configuration.get_topology())
        match self.configuration.get_topology():
            case constants.MASTER_WORKER | constants.PEER_TO_PEER | constants.CUSTOM:
                dff_run_command.extend([str(int(self.configuration.get_force_cpu())),
                                        str(self.configuration.get_rounds()),
                                        str(self.configuration.get_epochs()),
                                        self.dataset.get_data_path(),
                                        str(self.json.get_clients_number()),
                                        self.configuration.get_torchscript_path()])
            case constants.EDGE_INFERENCE:
                dff_run_command.extend([str(int(self.configuration.get_force_cpu())),
                                        self.dataset.get_data_path(),
                                        str(self.json.get_clients_number()),
                                        "1",  # TODO: Add support for multiple groups
                                        self.configuration.get_torchscript_path()])
            case constants.MVDET:
                dff_run_command.extend([str(int(self.configuration.get_force_cpu())),
                                        self.dataset.get_data_path(),
                                        str(self.json.get_clients_number()),
                                        "1"])  # TODO: Add support for multiple groups
        return dff_run_command

    def kill(self) -> List[int]:
        """Method to force-kill the existing FastFederatedLearning processes.
        It exploits multiple SSH connections to run pkill commands on all provided hosts.

        :return: list of the exit codes received by the contacted hosts.
        :rtype: List[int]
        """
        self.logger.info("Killing all FastFL jobs on the following hosts: %s", self.json.get_hosts())
        client = ParallelSSHClient(self.json.get_hosts())
        output = client.run_command('pkill -f -9 FastFederatedLearning')

        exit_codes: List[int] = []
        for host_out in output:
            for line in host_out.stdout:
                self.logger.info("Message received from host %s: %s", host_out, line)
            exit_codes.append(host_out.exit_code)
        self.logger.debug("Exit codes received from the hosts: %s", dict(zip(self.json.get_hosts(), exit_codes)))
        self.logger.info("Killing FastFederatedLearning process done.")

        return exit_codes
