import constants
from configuration import Configuration
from experiment import Experiment
from model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG_PATH = "/mnt/shared/gmittone/FastFederatedLearning/workspace/config.json"
DFF_RUN_PATH = "/mnt/shared/gmittone/FastFederatedLearning/libs/fastflow/ff/distributed/loader/dff_run"
DATA_PATH = "/mnt/shared/gmittone/FastFederatedLearning/data"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


compiled_model = Model(Net()).compile(torch.rand(1, 1, 28, 28))

config = Configuration(json_path=CONFIG_PATH, data_path=DATA_PATH, runner_path=DFF_RUN_PATH,
                       endpoints=["medium-0" + str(rank) + ":800" + str(rank) for rank in range(1, 6)],
                       topology=constants.MASTER_WORKER)
experiment = Experiment(config, model=compiled_model)

experiment.kill()
experiment.run_experiment()
