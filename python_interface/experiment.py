import torch
from torch.jit import ScriptModule, save
from configuration import Configuration
from json_generator import FFjson
from subprocess import call
from pssh.clients.ssh.parallel import ParallelSSHClient


class Experiment:

    def __init__(self, configuration: Configuration, model: ScriptModule):
        self.configuration: Configuration = configuration
        self.json: FFjson = self.configuration.get_json()
        self.model: ScriptModule = model

    def run_experiment(self):
        torch.jit.save(self.model, self.configuration.get_torchscript_path())
        self.json.generate_json_file(self.configuration.get_json_path())
        call([self.configuration.get_runner_path(),
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
              self.configuration.get_torchscript_path()])

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
