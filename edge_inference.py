from python_interface.DSL.flgraph import *
from python_interface.DSL.flgraph.flgraph import *
from python_interface.configuration import Configuration
from python_interface.dataset import Dataset
from python_interface.experiment import Experiment
from python_interface.utils.constants import EDGE_INFERENCE

FFL_DIR = "/mnt/shared/gmittone/FastFederatedLearning/"
DATA_PATH = FFL_DIR + "data/Ranger_Roll_m.mp4"
MODEL_PATH = FFL_DIR + "data/yolov5n.torchscript"
nodes = ["small-0" + str(rank) + ":800" + str(rank) for rank in range(1, 6)]

# + ["medium-0" + str(rank) + ":800" + str(rank) for rank in range(1, 10)] \
# + ["medium-" + str(rank) + ":800" + str(rank) for rank in range(10, 21)] \
# + ["large-0" + str(rank) + ":800" + str(rank) for rank in range(1, 6)]

ff_executable = FLGraph([
    Wrapper("Initialisation_inference"),  # TODO: add command line parameters here
    Parallel([
        Wrapper("Inference"),
        Reduce("Father"),
        Wrapper("Combine")
    ]),
    Reduce("Father"),
    Wrapper("Control_room")
]).compile(opencv_required=True)

config = Configuration(endpoints=nodes, executable_path=ff_executable, torchscript_path=MODEL_PATH,
                       topology=EDGE_INFERENCE)
dataset = Dataset(DATA_PATH)  # TODO: add dataset support
experiment = Experiment(config, dataset=dataset)

# experiment.kill()
experiment.run_experiment()
