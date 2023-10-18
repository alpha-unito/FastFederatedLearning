"""
Class responsible for Reduce structures
"""

from typing import List, TextIO, Final

import python_interface.DSL.flgraph as flgraph
from .building_block import BuildingBlock

add_aggregator: Final[str] = """if (groupName.compare(loggerName) == 0)
        std::cout << "Aggreator creation..." << std::endl;
    Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
    FedAvg <StateDict> aggregator(*net->state_dict());
    ff_node *federator = new ff_comb(new MiNodeAdapter<StateDict>,
                                     new Federator(net->state_dict(), net, aggregator, num_workers, rounds));
    a2a.add_firstset<ff_node>({federator});
    a2a.createGroup("W0") << federator;
    """


class Reduce(BuildingBlock):
    """ Class responsible for Reduce structures """

    def __init__(self, strategy: str = "FedAvg"):  # TODO: create aggregation strategies constants
        """ Initialisation of a Reduce Building Block.

        :param strategy: type of strategy to use for the reduce operation.
        :type strategy: str
        """
        super().__init__(self.__str__())

        self.strategy: str = strategy

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIO):
        """ Compilation of the Reduce Building Block.

        :param building_blocks: remaining Building Blocks to compile.
        :type building_blocks: List[BuildingBlock]
        :param source_file: C/C++ source file to write on.
        :type source_file: TextIO
        """
        if building_blocks:
            first_bb: BuildingBlock
            remaining_bb: List[BuildingBlock]
            first_bb, *remaining_bb = building_blocks
            if isinstance(first_bb, flgraph.Broadcast):
                source_file.write(add_aggregator)
                building_blocks.pop()
            self.logger.debug("Analysing the %s building block...", first_bb)
            first_bb.compile(remaining_bb, source_file)
