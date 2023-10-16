"""
Class responsible for feedback loop structures
"""
from typing import List, TextIO

from .building_block import BuildingBlock


class Feedback(BuildingBlock):
    """ Class responsible for feedback loop structures """

    def __init__(self, tasks: List[BuildingBlock], rounds: int = 1):
        """ Initialisation of the Feedback Building Block.

        :param tasks: nested Building Blocks to analyse.
        :type tasks: List[BuildingBlock]
        :param rounds: number of times to run the feedback loop.
        :type rounds: int
        """
        super().__init__(self.__str__())

        self.tasks: List[BuildingBlock] = tasks
        self.rounds: int = rounds

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIO):
        """ Compilation of the Feedback Building Block

        :param building_blocks: remaining Building Blocks to compile.
        :type building_blocks: List[BuildingBlock]
        :param source_file: C/C++ source file to write on.
        :type source_file: TextIO
        """
        if self.tasks:
            first_bb: BuildingBlock
            remaining_bb: List[BuildingBlock]
            first_bb, *remaining_bb = self.tasks
            self.logger.debug("Analysing the %s task...", first_bb)
            first_bb.compile(remaining_bb, source_file)
        if building_blocks:
            first_bb: BuildingBlock
            remaining_bb: List[BuildingBlock]
            first_bb, *remaining_bb = building_blocks
            self.logger.debug("Analysing the %s building block...", first_bb)
            first_bb.compile(remaining_bb, source_file)
