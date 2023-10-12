import torch.nn as nn
import torch.nn.functional as F

from python_interface.DSL.flgraph import *
from python_interface.DSL.flgraph.flgraph import *

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


federation = FLGraph([
    Initialisation(),  # TODO: add command line parameters here
    Feedback([
        Parallel([
            Wrapper("Train"),
            Wrapper("Test")
        ]),
        Reduce(),
        Broadcast(),
    ])
])

federation.compile()
federation.run()
