"""
Class responsible for Parallel structures
"""
from typing import List, Final, TextIO

from .building_block import BuildingBlock
from .wrapper import Wrapper

init_code: Final[str] = """
    if (groupName.compare(loggerName) == 0)
        std::cout << "Worker creation..." << std::endl;
    ff_a2a a2a;
    std::vector < ff_node * > w;
    for (int i = 1; i <= num_workers; ++i) {
        Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
        auto optimizer = std::make_shared<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001));
"""

end_code: Final[str] = """
    }
    a2a.add_secondset(w);"""

worker_creation: Final[str] = """
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
    """


class Parallel(BuildingBlock):
    """ Class responsible for Parallel structures """

    def __init__(self, tasks: List[BuildingBlock], replicas: int = 1):
        """ Initialisation of the Parallel Building Block.

        :param tasks: nested Building Blocks to analyse.
        :type tasks: List[BuildingBlock]
        :param replicas: number of different replicas to run.
        :type replicas: int
        """
        super().__init__(self.__str__())

        self.tasks: List[BuildingBlock] = tasks
        self.replicas: int = replicas

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIO):
        """ Compilation of the Parallel Building Block

        :param building_blocks: remaining Building Blocks to compile.
        :type building_blocks: List[BuildingBlock]
        :param source_file: C/C++ source file to write on.
        :type source_file: TextIO
        """
        source_file.write(init_code)
        if self.tasks:
            first_bb: BuildingBlock
            remaining_bb: List[BuildingBlock]
            first_bb, *remaining_bb = self.tasks
            self.logger.debug("Analysing the %s task...", first_bb)
            first_bb.compile(remaining_bb, source_file)
        if all(isinstance(task, Wrapper) for task in self.tasks):
            if all(task.get_label() in ["Train", "Test"] for task in self.tasks):
                source_file.write(worker_creation)
        source_file.write(end_code)
        if building_blocks:
            first_bb: BuildingBlock
            remaining_bb: List[BuildingBlock]
            first_bb, *remaining_bb = building_blocks
            self.logger.debug("Analysing the %s building block...", first_bb)
            first_bb.compile(remaining_bb, source_file)
