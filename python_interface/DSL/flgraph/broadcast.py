"""
Class responsible for Broadcasting structures
"""
from typing import List, TextIO, Final

from .building_block import BuildingBlock


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
        pass
