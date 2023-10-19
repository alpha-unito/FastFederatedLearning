"""
FastFlow Sequential and Parallel code wrapper
"""
from typing import Final, List, TextIO

from .building_block import BuildingBlock
from .feedback import Feedback

init_code: Final[str] = """#include <mutex>
#include <iostream>

#include <ff/dff.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "C/dfl/federation.hpp"
#include "C/dfl/fedavg.hpp"
#include "C/utils/net.hpp"
#include "C/utils/utils.hpp"
#include "C/utils/serialize.hpp"

using namespace ff;

template<typename T>
struct MiNodeAdapter : ff_minode_t<T> {
    T *svc(T *in) { return in; }
};

template<typename T>
void serializefreetask(T *o, StateDict *input) {}

int main(int argc, char *argv[]) {
    timer chrono = timer("Total execution time");

    std::string groupName = "W0";
    const std::string loggerName = "W0";

#ifndef DISABLE_FF_DISTRIBUTED
    for (int i = 0; i < argc; i++)
        if (strstr(argv[i], "--DFF_GName") != NULL) {
            char *equalPosition = strchr(argv[i], '=');
            groupName = std::string(++equalPosition);
            break;
        }
    if (DFF_Init(argc, argv) < 0) {
        error("Error while executing: DFF_Init");
        return -1;
    }
#endif

    int num_workers{3};         // Number of workers
    int train_batchsize{64};    // Train batch size
    int test_batchsize{64};     // Test batch size
    int train_epochs{2};        // Number of training epochs at workers in each round
    int rounds{10};             // Number of training rounds
    int forcecpu{0};            // Force the execution on the CPU
    int nt{4};                  // Number of threads per process
    char *data_path;            // Patch to the dataset (absolute or with respect to build directory)
    char *inmodel;              // Path to a TorchScript representation of a DNN

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
            if (groupName.compare(loggerName) == 0)
                std::cout
                        << "Usage: masterworker [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path] [num_workers] [model_path]" << std::endl;
            exit(0);
        } else
            forcecpu = atoi(argv[1]);
    }
    if (argc >= 3)
        rounds = atoi(argv[2]);
    if (argc >= 4)
        train_epochs = atoi(argv[3]);
    if (argc >= 5)
        data_path = argv[4];
    if (argc >= 6)
        num_workers = atoi(argv[5]);
    if (argc >= 7)
        inmodel = argv[6];
    if (groupName.compare(loggerName) == 0)
        std::cout << "Training on " << num_workers << " workers." << std::endl;

    torch::DeviceType device_type;
    if (torch::cuda::is_available() && !forcecpu) {
        if (groupName.compare(loggerName) == 0)
            std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        if (groupName.compare(loggerName) == 0)
            std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    torch::cuda::manual_seed_all(42);
    
    if (groupName.compare(loggerName) == 0)
        std::cout << "Data loading..." << std::endl;
    auto train_dataset = torch::data::datasets::MNIST(data_path).map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::MNIST(data_path, torch::data::datasets::MNIST::Mode::kTest).map(
            torch::data::transforms::Stack<>());
"""

end_code_feedback: Final[str] = """
#ifdef DISABLE_FF_DISTRIBUTED
    a2a.wrap_around();
    if (a2a.run_and_wait_end() < 0) {
        error("Error while executing: All-to-All");
        return -1;
    }
#else
    ff::ff_pipeline pipe;
    pipe.add_stage(&a2a);
    pipe.wrap_around();
    if (pipe.run_and_wait_end() < 0) {
        error("Error while executing: Pipe");
        return -1;
    }
#endif

    if (groupName.compare(loggerName) == 0)
        chrono.stop();
    return 0;
}
"""

end_code_no_feedback: Final[str] = """
#ifdef DISABLE_FF_DISTRIBUTED
    a2a.wrap_around();
    if (a2a.run_and_wait_end() < 0) {
        error("Error while executing: All-to-All");
        return -1;
    }
#else
    ff::ff_pipeline pipe;
    pipe.add_stage(&a2a);
    pipe.wrap_around();
    if (pipe.run_and_wait_end() < 0) {
        error("Error while executing: Pipe");
        return -1;
    }
#endif

    if (groupName.compare(loggerName) == 0)
        chrono.stop();
    return 0;
}
"""

worker_creation_ms: Final[str] = """
        ff_node *worker = new ff_comb(new MiNodeAdapter<StateDict>,
                                      new Worker(i, net, net->state_dict(), train_epochs, optimizer,
                                                 torch::data::make_data_loader(train_dataset,
                                                                               torch::data::samplers::DistributedRandomSampler(
                                                                                       train_dataset.size().value(),
                                                                                       num_workers,
                                                                                       i % num_workers,
                                                                                       true),
                                                                               train_batchsize),
                                                 torch::data::make_data_loader(test_dataset, test_batchsize),
                                                 device));
        w.push_back(worker);
        a2a.createGroup("W" + std::to_string(i)) << worker;
    """

worker_creation_p2p: Final[str] = """
        ff_node *peer = new ff_comb(new MiNodeAdapter<StateDict>(),
                                            new Peer(i, local_net, local_net->state_dict(), train_epochs, optimizer,
                                                     torch::data::make_data_loader(train_dataset,
                                                                                   torch::data::samplers::DistributedRandomSampler(
                                                                                           train_dataset.size().value(),
                                                                                           num_workers,
                                                                                           i % num_workers,
                                                                                           true),
                                                                                   train_batchsize),
                                                     fed_net, fed_net->state_dict(), aggregator, num_workers, rounds,
                                                     torch::data::make_data_loader(test_dataset, test_batchsize),
                                                     device), true, true);
        left.push_back(peer);
    """


class Wrapper(BuildingBlock):
    """ FastFlow Sequential and Parallel code wrapper """

    def __init__(self, label: str):
        """ Initialisation of a Wrapper Building Block.

        :param label: label assigned to the Wrapper; supported values: Train, Test
        :type label: string
        """
        super().__init__(self.__class__.__name__)

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
        if building_blocks:
            match self.label:
                case "Initialisation":
                    source_file.write(init_code)
                    if building_blocks:
                        first_bb: BuildingBlock
                        remaining_bb: List[BuildingBlock]
                        first_bb, *remaining_bb = building_blocks
                        self.logger.debug("Analysing the %s building block...", first_bb)
                        first_bb.compile(remaining_bb, source_file)
                        if isinstance(first_bb, Feedback):
                            source_file.write(end_code_feedback)
                        else:
                            source_file.write(end_code_no_feedback)
                case "Train":
                    first_bb: BuildingBlock
                    remaining_bb: List[BuildingBlock]
                    first_bb, *remaining_bb = building_blocks
                    if isinstance(first_bb, Wrapper) and first_bb.get_label() == "Test":
                        if remaining_bb:
                            source_file.write(worker_creation_p2p)
                        else:
                            source_file.write(worker_creation_ms)
                    self.logger.debug("Analysing the %s task...", first_bb)
                    first_bb.compile(remaining_bb, source_file)
                case "Test":
                    first_bb: BuildingBlock
                    remaining_bb: List[BuildingBlock]
                    first_bb, *remaining_bb = building_blocks
                    self.logger.debug("Analysing the %s task...", first_bb)
                    first_bb.compile(remaining_bb, source_file)

    def __str__(self) -> str:
        """
        Get the caller's class name

        :return: name of the calling class.
        :rtype: str
        """
        return self.get_label()
