import os, tempfile

import torch
from torch.jit import ScriptModule, save
from configuration import Configuration
from json_generator import FFjson
from subprocess import call
from pssh.clients.ssh.parallel import ParallelSSHClient

CONFIG_PATH = "/mnt/shared/gmittone/FastFederatedLearning/workspace/config.json"
TORCHSCRIPT_PATH = "/mnt/shared/gmittone/FastFederatedLearning/workspace/model.pt"


class Experiment:

    def __init__(self, configuration: Configuration, model: ScriptModule):
        self.configuration: Configuration = configuration
        self.json: FFjson = self.configuration.get_json()
        self.model: ScriptModule = model

    def run_experiment(self):
        torch.jit.save(self.model, "/mnt/shared/gmittone/FastFederatedLearning/workspace/model.pt")
        self.json.generate_json_file(CONFIG_PATH)
        call([self.configuration.get_runner_path(), "-V", "-p", "TCP", "-f ", self.configuration.get_json_path(),
              self.configuration.get_executable_path(), "1", "1", "1", self.configuration.get_data_path(),
              str(self.json.get_clients_number())])

    def kill(self) -> int:
        client = ParallelSSHClient(self.json.get_hosts())
        output = client.run_command('pkill -f -9 FastFederatedLearning')
        for host_out in output:
            for line in host_out.stdout:
                print(line)
            exit_code = host_out.exit_code
        return exit_code

    def stream_metrics(self):
        pass
