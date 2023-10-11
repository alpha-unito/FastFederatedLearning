"""
Class responsible for feedback loop structures
"""
from io import TextIOWrapper
from typing import List

from .building_block import BuildingBlock

init_code: str = """if (groupName.compare(loggerName) == 0)
        std::cout << "Worker creation..." << std::endl;
    ff_a2a a2a;
    std::vector < ff_node * > w;
    for (int i = 1; i <= num_workers; ++i) {
        Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
        auto optimizer = std::make_shared<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001));
"""

end_code: str = """        w.push_back(worker);
        a2a.createGroup("W" + std::to_string(i)) << worker;
    }
    a2a.add_secondset(w);"""


class Feedback(BuildingBlock):

    def __init__(self, tasks: List[BuildingBlock], rounds: int = 1):
        super().__init__(self.__class__.__name__)
        self.tasks: List[BuildingBlock] = tasks  # TODO: create a type for all components
        self.rounds: int = rounds

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIOWrapper):
        source_file.write(init_code)
        self.tasks[0].compile(self.tasks[1:], source_file)
        source_file.write(end_code)
        building_blocks[0].compile(building_blocks[1:], source_file)
