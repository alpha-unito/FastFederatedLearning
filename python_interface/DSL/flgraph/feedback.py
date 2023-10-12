"""
Class responsible for feedback loop structures
"""
from io import TextIOWrapper
from typing import List

from .building_block import BuildingBlock


class Feedback(BuildingBlock):

    def __init__(self, tasks: List[BuildingBlock], rounds: int = 1):
        super().__init__(self.__class__.__name__)
        self.tasks: List[BuildingBlock] = tasks  # TODO: create a type for all components
        self.rounds: int = rounds

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIOWrapper):
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
