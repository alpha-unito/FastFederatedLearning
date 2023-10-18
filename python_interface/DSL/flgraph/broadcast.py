"""
Class responsible for Broadcasting structures
"""
from typing import List, TextIO, Final

import python_interface.DSL.flgraph as flgraph
from .building_block import BuildingBlock

add_distributor: Final[str] = """ff::ff_node *distributor = new ff::ff_comb(new MiNodeAdapter<StateDict>(),
                                                   new Distributor<StateDict>(i, num_workers, device), true, true);
        right.push_back(distributor);
        a2a.createGroup("W" + std::to_string(i)) << peer << distributor;
    """


class Broadcast(BuildingBlock):
    """ Class responsible for Broadcasting structures """

    def __init__(self, policy: str = "broadcast"):  # TODO: create aggregation strategies constants
        """ Initialisation of a Broadcast Building Block.

        :param policy: policy to adopt for the broadcast operation.
        :type policy: string
        """
        super().__init__(self.__str__())

        self.policy: Final[str] = policy

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIO):
        """ Compilation of the Broadcast Building Block

        :param building_blocks: remaining Building Blocks to compile.
        :type building_blocks: List[BuildingBlock]
        :param source_file: C/C++ source file to write on.
        :type source_file: TextIO
        """
        if building_blocks:
            first_bb: BuildingBlock
            remaining_bb: List[BuildingBlock]
            first_bb, *remaining_bb = building_blocks
            if isinstance(first_bb, flgraph.Reduce):
                source_file.write(add_distributor)
                building_blocks.pop()
            self.logger.debug("Analysing the %s building block...", first_bb)
            first_bb.compile(remaining_bb, source_file)
