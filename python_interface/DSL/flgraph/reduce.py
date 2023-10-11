"""
Class responsible for Reduce structures
"""

from io import TextIOWrapper
from typing import List

from .broadcast import Broadcast
from .building_block import BuildingBlock

add_aggregator: str = """if (groupName.compare(loggerName) == 0)
        std::cout << "Aggreator creation..." << std::endl;
    Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
    FedAvg <StateDict> aggregator(*net->state_dict());
    ff_node *federator = new ff_comb(new MiNodeAdapter<StateDict>,
                                     new Federator(net->state_dict(), net, aggregator, num_workers, rounds));
    a2a.add_firstset<ff_node>({federator});
    a2a.createGroup("W0") << federator;
    """


class Reduce(BuildingBlock):

    def __init__(self, strategy: str = "FedAvg"):  # TODO: create aggregation strategies constants
        super().__init__(self.__class__.__name__)
        self.strategy: str = strategy

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIOWrapper):
        if isinstance(building_blocks[0], Broadcast):
            source_file.write(add_aggregator)
        building_blocks[1].compile(building_blocks[2:], source_file)
