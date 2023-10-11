"""
Class responsible for Broadcasting structures
"""
from io import TextIOWrapper
from typing import List

from .building_block import BuildingBlock


class Broadcast(BuildingBlock):

    def __init__(self, policy: str = "broadcast"):  # TODO: create aggregation strategies constants
        super().__init__(self.__class__.__name__)
        self.policy: str = policy

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIOWrapper):
        pass
