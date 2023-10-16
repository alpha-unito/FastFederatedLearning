"""
This class provides the interface for creating, compiling and running a Federated Learning graph
"""

import logging
from subprocess import call, PIPE, Popen, STDOUT
from typing import List

from python_interface.custom.custom_types import PathLike
from python_interface.utils.constants import DEFAULT_WORKSPACE_DIR, DEFAULT_LIBS_DIR, DEFAULT_FFL_DIR
from python_interface.utils.utils import check_and_create_path, get_logger
from .building_block import BuildingBlock


class FLGraph:
    """ This class provides the interface for creating, compiling and running a Federated Learning graph """

    def __init__(self, tasks: List[BuildingBlock]):
        """Class modeling a graph representing a Federated Learning tasks.

           :param tasks: List of tasks to be carried out by the federated infrastructure; they implicitly specify the
                FL graph structure.
           :type tasks: List[Compute, Repeat, Distribute, Aggregate]
       """
        self.logger: logging.Logger = get_logger(self.__class__.__name__)

        self.tasks: List[BuildingBlock] = tasks

    def compile(self, workspace: PathLike = DEFAULT_WORKSPACE_DIR) -> PathLike:
        """
        Translation of the provided structure into a concrete C/C++ source file implementing the specified structure.
        The source file is then compiled and linked.

        :param workspace: path to the workspace folder.
        :type workspace: PathLike
        :return: path to the executable file.
        :rtype: PathLike
        """
        self.logger.info("Starting the creation of the FastFlow C/C++ source file...")
        check_and_create_path(workspace, "C source file", self.logger)
        source_path: PathLike = workspace + "source.cpp"
        source_file = open(source_path, "w")
        if self.tasks:
            first_bb: BuildingBlock
            remaining_bb: List[BuildingBlock]
            first_bb, *remaining_bb = self.tasks
            self.logger.debug("Analysing the %s building block...", first_bb)
            first_bb.compile(remaining_bb, source_file)
        source_file.close()

        # TODO: add error handling
        self.logger.info("FastFlow C/C++ source file created at %s. Starting compilation...", source_path)
        with Popen(["/usr/bin/c++",
                    "-DNO_DEFAULT_MAPPING",
                    "-DUSE_C10D_GLOO",
                    "-DUSE_DISTRIBUTED",
                    "-DUSE_RPC",
                    "-DUSE_TENSORPIPE",
                    "-I" + DEFAULT_LIBS_DIR + "fastflow",
                    "-I" + DEFAULT_LIBS_DIR + "cereal/include",
                    "-I" + DEFAULT_FFL_DIR + "..",
                    "-I" + DEFAULT_FFL_DIR,
                    "-isystem", DEFAULT_LIBS_DIR + "torch/include",
                    "-isystem", DEFAULT_LIBS_DIR + "torch/include/torch/csrc/api/include",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    "-pthread",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    "-MD",
                    "-MT", workspace + "source.cpp.o",
                    "-MF", workspace + "source.cpp.o.d",
                    "-o", workspace + "source.cpp.o",
                    "-c", workspace + "source.cpp"], stdout=PIPE, stderr=STDOUT) as process:
            for line in process.stdout:
                self.logger.critical(line.decode('utf8'))

        self.logger.info("Compilation completed successfully. Starting linking...")
        with Popen(["/usr/bin/c++",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    "-pthread",
                    "-rdynamic",
                    "-Wl,-rpath",
                    "-Wl,/usr/local/lib",
                    "-Wl,--enable-new-dtags",
                    workspace + "source.cpp.o",
                    "-o", workspace + "source",
                    "-Wl,-rpath," + DEFAULT_LIBS_DIR + "torch/lib:/usr/local/lib",
                    DEFAULT_LIBS_DIR + "torch/lib/libtorch.so",
                    DEFAULT_LIBS_DIR + "torch/lib/libc10.so",
                    DEFAULT_LIBS_DIR + "torch/lib/libkineto.a",
                    "-Wl,--no-as-needed," + DEFAULT_LIBS_DIR + "torch/lib/libtorch_cpu.so",
                    "-Wl,--as-needed",
                    DEFAULT_LIBS_DIR + "torch/lib/libc10.so",
                    "-Wl,--no-as-needed," + DEFAULT_LIBS_DIR + "torch/lib/libtorch.so",
                    "-Wl,--as-needed",
                    "/usr/local/lib/libmpi.so"], stdout=PIPE, stderr=STDOUT) as process:
            for line in process.stdout:
                self.logger.critical(line.decode('utf8'))

        self.logger.info("Linkingn completed successfully. Removing build files...")
        call(["rm", "-rf", workspace + "source.cpp.o", workspace + "source.cpp.o.d"])

        return workspace + "source"
