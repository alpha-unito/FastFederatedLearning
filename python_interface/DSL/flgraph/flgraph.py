"""
This class provides the interface for creating, compiling and running a Federated Learning graph
"""

import logging
from typing import List

from python_interface.custom.custom_types import PathLike
from python_interface.utils.constants import DEFAULT_WORKSPACE_DIR
from python_interface.utils.utils import check_and_create_path, get_logger


class FLGraph:

    def __init__(self, tasks: List):  # TODO: add type
        """Class modeling a graph representing a Federated Learning tasks.

           :param nodes: List of nodes involved in the FL task, specified as hostnames, ip, or ip:port.
           :type nodes: Union[str, List[str]]
           :param tasks: List of tasks to be carried out by the federated infrastructure; they implicitly specify the
                FL graph structure.
           :type tasks: List[Compute, Repeat, Distribute, Aggregate]
           :param coordinator: specify the coordination node, if any.
           :type coordinator: Optional[str]
       """
        self.logger: logging.Logger = get_logger(self.__class__.__name__)

        self.tasks: List = tasks

    def run(self):
        return None

    def compile(self, workspace: PathLike = DEFAULT_WORKSPACE_DIR):
        check_and_create_path(workspace, "C source file", self.logger)
        source_path: PathLike = workspace + "source.c"
        source_file = open(source_path, "w")
        self.tasks[0].compile(self.tasks[1:], source_file)
        source_file.close()
        # TODO: add compilation
