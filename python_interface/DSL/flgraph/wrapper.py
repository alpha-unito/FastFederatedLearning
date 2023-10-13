"""
FastFlow Sequential and Parallel code wrapper
"""
from typing import Final, List, TextIO

from .building_block import BuildingBlock


class Wrapper(BuildingBlock):
    """ FastFlow Sequential and Parallel code wrapper """

    def __init__(self, label: str):
        """ Initialisation of a Wrapper Building Block.

        :param label: label assigned to the Wrapper; supported values: Train, Test
        :type label: string
        """
        super().__init__(self.__str__())

        self.label: Final[str] = label

    def get_label(self) -> str:
        """ Getter for the Wrapper Building Block label.

        :return: wrapper's label
        :rtype: str
        """
        return self.label

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIO):
        """ Compilation of the Wrapper Building Block

        :param building_blocks: remaining Building Blocks to compile.
        :type building_blocks: List[BuildingBlock]
        :param source_file: C/C++ source file to write on.
        :type source_file: TextIO
        """
        pass
