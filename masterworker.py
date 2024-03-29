import torch
import torch.nn as nn
import torch.nn.functional as F

from python_interface.DSL.flgraph import *
from python_interface.DSL.flgraph.flgraph import *
from python_interface.configuration import Configuration
from python_interface.dataset import Dataset
from python_interface.experiment import Experiment
from python_interface.model import Model

FFL_DIR = "/mnt/shared/gmittone/FastFederatedLearning/"
DATA_PATH = FFL_DIR + "data/"
nodes = ["small-0" + str(rank) + ":800" + str(rank) for rank in range(1, 6)]


# + ["medium-0" + str(rank) + ":800" + str(rank) for rank in range(1, 10)] \
# + ["medium-" + str(rank) + ":800" + str(rank) for rank in range(10, 21)] \
# + ["large-0" + str(rank) + ":800" + str(rank) for rank in range(1, 6)]


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


ff_executable = FLGraph([
    Wrapper("Initialisation"),  # TODO: add command line parameters here
    Feedback([
        Parallel([
            Wrapper("Train"),
            Wrapper("Test")
        ]),
        Reduce("FedAvg"),
        Broadcast(),
    ])
]).compile()

config = Configuration(endpoints=nodes, executable_path=ff_executable)
model = Model(model=Net(), example=torch.rand(128, 1, 28, 28), optimize=False)
dataset = Dataset(DATA_PATH)  # TODO: add dataset support
experiment = Experiment(config, model=model, dataset=dataset)

# experiment.kill()
experiment.run_experiment()
