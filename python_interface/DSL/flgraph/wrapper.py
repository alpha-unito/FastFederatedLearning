from io import TextIOWrapper
from typing import Final, List

from .building_block import BuildingBlock


class Wrapper(BuildingBlock):

    def __init__(self, label: str):
        super().__init__(self.__class__.__name__)
        self.label: Final[str] = label

    def get_label(self) -> str:
        return self.label

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIOWrapper):
        pass
