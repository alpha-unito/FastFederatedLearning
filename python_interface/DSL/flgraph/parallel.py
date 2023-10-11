"""
Class responsible for Parallel structures
"""
from io import TextIOWrapper
from typing import List

from .building_block import BuildingBlock
from .wrapper import Wrapper

worker_creation = """
    if (groupName.compare(loggerName) == 0)
        std::cout << "Worker creation..." << std::endl;
    ff_a2a a2a;
    std::vector < ff_node * > w;
    for (int i = 1; i <= num_workers; ++i) {
        Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
        auto optimizer = std::make_shared<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001));

        ff_node *worker = new ff_comb(new MiNodeAdapter<StateDict>,
                                      new Worker(i, net, net->state_dict(), train_epochs, optimizer,
                                                 torch::data::make_data_loader(train_dataset,
                                                                               torch::data::samplers::DistributedRandomSampler(
                                                                                       train_dataset.size().value(),
                                                                                       num_workers,
                                                                                       i % num_workers,
                                                                                       true),
                                                                               train_batchsize),
                                                 torch::data::make_data_loader(test_dataset, test_batchsize),
                                                 device));
        w.push_back(worker);
        a2a.createGroup("W" + std::to_string(i)) << worker;
    }
    a2a.add_secondset(w);
    """


class Parallel(BuildingBlock):

    def __init__(self, tasks: List[BuildingBlock], replicas: int = 1):
        super().__init__(self.__class__.__name__)
        self.tasks: List[BuildingBlock] = tasks
        self.replicas: int = replicas

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIOWrapper):
        if all(isinstance(task, Wrapper) for task in self.tasks):
            if all(task.get_label() in ["Train", "Test"] for task in self.tasks):
                source_file.write(worker_creation)
