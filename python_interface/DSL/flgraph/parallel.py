"""
Class responsible for Parallel structures
"""
from typing import List, Final, TextIO

from .building_block import BuildingBlock

init_code_ms: Final[str] = """
    if (groupName.compare(loggerName) == 0)
        std::cout << "Worker creation..." << std::endl;
    ff_a2a a2a;
    std::vector < ff_node * > w;
    num_workers--;
    for (int i = 1; i <= num_workers; ++i) {
        Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
        auto optimizer = std::make_shared<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001));
    """

end_code_ms: Final[str] = """
    }
    a2a.add_secondset(w);
    """

init_code_p2p: Final[str] = """
    if (groupName.compare(loggerName) == 0)
        std::cout << "Peers creation..." << std::endl;
    ff::ff_a2a a2a;
    std::vector < ff::ff_node * > left;
    std::vector < ff::ff_node * > right;
    for (int i = 0; i < num_workers; ++i) {
        Net <torch::jit::Module> *local_net = new Net<torch::jit::Module>(inmodel);
        auto optimizer = std::make_shared<torch::optim::Adam>(local_net->parameters(),
                                                              torch::optim::AdamOptions(0.001));
                                                              
    Net <torch::jit::Module> *fed_net = new Net<torch::jit::Module>(inmodel);
    FedAvg <StateDict> aggregator(*fed_net->state_dict());
    """

end_code_p2p: Final[str] = """
    }
    a2a.add_firstset(left);
    a2a.add_secondset(right);
    """

init_code_inference: Final[str] = """    if (groupName.compare(loggerName) == 0)
        std::cout << "Cameras creation..." << std::endl;
    ff_a2a a2a;
    std::vector < ff_node * > globalLeft;
    for (int i = 1; i <= num_workers; ++i) {
        ff_pipeline *pipe = new ff_pipeline;   // <---- To be removed and automatically added
        ff_a2a *local_a2a = new ff_a2a;
        pipe->add_stage(local_a2a, true);
        """

end_code_inference: Final[str] = """    
    }
    a2a.add_firstset(globalLeft, true);
    """


class Parallel(BuildingBlock):
    """ Class responsible for Parallel structures """

    def __init__(self, tasks: List[BuildingBlock]):
        """ Initialisation of the Parallel Building Block.

        :param tasks: nested Building Blocks to analyse.
        :type tasks: List[BuildingBlock]
        :param replicas: number of different replicas to run.
        :type replicas: int
        """
        super().__init__(self.__str__())

        self.tasks: List[BuildingBlock] = tasks

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIO):
        """ Compilation of the Parallel Building Block

        :param building_blocks: remaining Building Blocks to compile.
        :type building_blocks: List[BuildingBlock]
        :param source_file: C/C++ source file to write on.
        :type source_file: TextIO
        """
        if len(self.tasks) == 1 and str(self.tasks[0]) == "Initialisation":
            self.logger.debug("Analysing the %s task...", self.tasks[0])
            self.tasks[0].compile(building_blocks, source_file)
        else:
            if building_blocks:
                if str(building_blocks[0]) == "FedAvg":
                    source_file.write(init_code_ms)
                elif str(building_blocks[0]) == "Father":
                    source_file.write(init_code_inference)
            else:
                source_file.write(init_code_p2p)
            if self.tasks:
                first_bb: BuildingBlock
                remaining_bb: List[BuildingBlock]
                first_bb, *remaining_bb = self.tasks
                self.logger.debug("Analysing the %s task...", first_bb)
                first_bb.compile(remaining_bb, source_file)
            if building_blocks:
                if str(building_blocks[0]) == "FedAvg":
                    source_file.write(end_code_ms)
                elif str(building_blocks[0]) == "Father":
                    source_file.write(end_code_inference)
            else:
                source_file.write(end_code_p2p)
            if building_blocks:
                first_bb: BuildingBlock
                remaining_bb: List[BuildingBlock]
                first_bb, *remaining_bb = building_blocks
                self.logger.debug("Analysing the %s building block...", first_bb)
                first_bb.compile(remaining_bb, source_file)
