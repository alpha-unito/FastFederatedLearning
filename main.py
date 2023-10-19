import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from python_interface.configuration import Configuration
from python_interface.dataset import Dataset
from python_interface.experiment import Experiment
from python_interface.model import Model
from python_interface.utils import constants

logging.basicConfig(level=logging.DEBUG)

FFL_DIR = "/mnt/shared/gmittone/FastFederatedLearning/"
# TORCHSCRIPT_PATH = FFL_DIR + "data/yolov5n.torchscript"
# DATA_PATH = FFL_DIR + "data/Ranger_Roll_m.mp4"
# DATA_PATH = FFL_DIR + "data/mvdet_data"

DATA_PATH = FFL_DIR + "data/"


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


# model = torchvision.models.resnet18(num_classes=10)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

config = Configuration(backend=constants.TCP, force_cpu=True, rounds=1, epochs=1,
                       endpoints=["small-0" + str(rank) + ":800" + str(rank) for rank in range(1, 6)]
                       # ["medium-0" + str(rank) + ":800" + str(rank) for rank in range(1, 10)]
                       # + ["medium-" + str(rank) + ":800" + str(rank) for rank in range(10, 11)]
                       # + ["large-0" + str(rank) + ":800" + str(rank) for rank in range(1, 6)]
                       , topology=constants.MASTER_WORKER)

model = Model(model=Net(), example=torch.rand(128, 1, 28, 28), optimize=False)
dataset = Dataset(DATA_PATH)
experiment = Experiment(config, model=model, dataset=dataset)

experiment.kill()
# experiment.run_experiment()
