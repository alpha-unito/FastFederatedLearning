"""
Basic Abstract Class for all the RISC-pB2L building blocks
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Self, TextIO

from python_interface.utils.utils import get_logger


class BuildingBlock(ABC):
    """Basic Abstract Class for all the RISC-pB2L building blocks"""

    def __init__(self, class_name: str):
        """
        Initializer of the Building Block

        :param class_name: name of the calling class; used for istantiating the correct logger.
        :type class_name: str
        """
        self.logger: logging.Logger = get_logger(class_name)

    @abstractmethod
    def compile(self, building_blocks: List[Self], source_file: TextIO):
        """
        Method for generating the C code corresponding to the calling Building Block

        :param building_blocks: list of the remaining Building Blocks to process.
        :type building_blocks: List[BuildingBlock]
        :param source_file: file handler on which to write the C source code.
        :type source_file: TextIOWrapper
        """
        pass

    def __str__(self) -> str:
        """
        Get the caller's class name

        :return: name of the calling class.
        :rtype: str
        """
        return self.__class__.__name__
