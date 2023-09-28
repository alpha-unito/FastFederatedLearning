"""
Class responsible for organizing, running, and eventually shutting down, the federation.
"""
import torch
import logging

from torch.jit import ScriptModule
from subprocess import call
from pssh.clients.ssh.parallel import ParallelSSHClient
from typing import List
from python_interface.configuration import Configuration
from python_interface.json_generator import JSONGenerator
from python_interface.utils import utils


class Experiment:
    """Object handling the experiment"""

    def __init__(self, configuration: Configuration, model: ScriptModule):
        """Object serving as interface for handling the Federated Learning experiment.

        :param configuration: a federation configuration file.
        :type configuration: Configuration
        :param model: a compiled PyTorch Model.
        :type model: ScriptModule
        """
        self.logger: logging.Logger = utils.get_logger(self.__class__.__name__)
        self.configuration: Configuration = configuration
        self.json: JSONGenerator = self.configuration.get_json()
        self.model: ScriptModule = model

        self.logger.info("Experiment set up correctly.")

    def run_experiment(self):
        """Experiment start-up.
        Saves the TorchScript model and the JSON configuration file, and then calls the C/C++ backend executable.
        """
        self.logger.info("Saving TorchScript model to: %s", self.model.get_torchscript_path())
        torch.jit.save(self.model, self.model.get_torchscript_path())

        self.logger.info("Creating JSON configuration file to: %s", self.configuration.get_json_path())
        self.json.generate_json_file(self.configuration.get_json_path())

        dff_run_command: List[str] = [self.configuration.get_runner_path(),
                                      "-V",
                                      "-p",
                                      self.configuration.get_backend(),
                                      "-f ",
                                      self.configuration.get_json_path(),
                                      self.configuration.get_executable_path(),
                                      str(int(self.configuration.get_force_cpu())),
                                      str(self.configuration.get_rounds()),
                                      str(self.configuration.get_epochs()),
                                      self.configuration.get_data_path(),
                                      str(self.json.get_clients_number()),
                                      self.model.get_torchscript_path()]
        self.logger.info("Launching the FastFlow backend: %s", dff_run_command)
        self.logger.info('-' * 80)
        call(dff_run_command)
        self.logger.info('-' * 80)
        self.logger.info("Experiment completed correctly.")

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
        # TODO: check the exit codes
        return exit_codes

    def stream_metrics(self):
        pass
