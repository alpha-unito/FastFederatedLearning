/*
 * FastFlow concurrent network:
 *
 *  -----------------------------------------------------------
 * |  /<------------ a2a(0)----------->/                        |
 * |   -------------------------------                          |
 * |  |                               |                         |
 * |  |  Client ->|                   |                         |
 * |  |           | -> level1Gatherer | ->|                     |
 * |  |  Client ->|                   |   |                     |
 * |  |                               |   |                     |
 * |   -------------------------------    |                     |
 * |       ....                           | --> level0Gatherer  |
 * |                                      |                     |
 * |   -------------------------------    |                     |
 * |  |                               |   |                     |
 * |  |  Client ->|                   |   |                     |
 * |  |           | -> level1Gatherer | ->|                     |
 * |  |  Client ->|                   |                         |
 * |  |                               |                         |
 * |   -------------------------------                          |
 * |  /<------------ a2a(n)----------->/                        |
 * |                                                            |
 *  ------------------------------------------------------------
 * /<-------------------------- mainA2A ----------------------->/
 *
 *
 * distributed version:
 *
 *  each a2a(i) is a group, a2a(i) --> Gi i>0
 *  level0Gatherer: G0
 *
 */

#include <iostream>

#include <ff/dff.hpp>
#include <torch/torch.h>

#include "C/utils/net.hpp"
#include "C/utils/utils.hpp"
#include "C/utils/serialize.hpp"
#include "C/dfl/video_inference.hpp"

using namespace ff;

//template<typename T>
//void serializefreetask(T *o, Frame *input) {}

int main(int argc, char *argv[]) {
    timer chrono = timer("Total execution time");

    std::string groupName = "W0";
    const std::string loggerName = "W0";

#ifndef DISABLE_FF_DISTRIBUTED
    for (int i = 0; i < argc; i++)
        if (strstr(argv[i], "--DFF_GName") != NULL) {
            char *equalPosition = strchr(argv[i], '=');
            groupName = std::string(++equalPosition);
            continue;
        }
    if (DFF_Init(argc, argv) < 0) {
        error("Error while executing: DFF_Init\n");
        return -1;
    }
#endif

    int num_workers{3};     // Number of workers
    int num_groups{1};      // Number of workers per group
    int forcecpu{0};        // Force the execution on the CPU
    char *data_path;        // Path to the dataset (absolute or with respect to build directory)
    char *inmodel;          // Path to a TorchScript representation of a DNN

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
            if (groupName.compare(loggerName) == 0)
                std::cout
                        << "Usage: edgeinference [forcecpu=0/1] [data_path] [groups=3] [clients/group=1] [model_path]\n";
            exit(0);
        } else
            forcecpu = atoi(argv[1]);
    }
    if (argc >= 3)
        data_path = argv[2];
    if (argc >= 4)
        num_workers = atoi(argv[3]);
    if (argc >= 5)
        num_groups = atoi(argv[4]);
    if (argc >= 6)
        inmodel = argv[5];
    if (groupName.compare(loggerName) == 0)
        std::cout << "Infering on " << num_workers - 1 << " cameras." << std::endl;

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
        std::cout << "Checking data existence..." << std::endl;
    if (!std::filesystem::exists(data_path)) {
        error("Video file cannot be found at path: %s\n", data_path);
        return -1;
    }

    if (groupName.compare(loggerName) == 0)
        std::cout << "Cameras creation..." << std::endl;
    ff_a2a a2a;
    std::vector < ff_node * > globalLeft;
    for (int i = 1; i <= num_workers; ++i) {
        ff_pipeline *pipe = new ff_pipeline;   // <---- To be removed and automatically added
        ff_a2a *local_a2a = new ff_a2a;
        pipe->add_stage(local_a2a, true);
        std::vector < ff_node * > localLeft;
        for (int j = 0; j < num_groups; ++j) {
            Net <torch::jit::Module> *model = new Net<torch::jit::Module>(inmodel);
            localLeft.push_back(
                    new EdgeNode < Net < torch::jit::Module > *, edgeMsg_t > (
                            "W(" + std::to_string(i) + "," + std::to_string(j) + ")", model, data_path));
        }
        local_a2a->add_firstset(localLeft, 0, true);
        local_a2a->add_secondset<ff_comb>(
                {new ff_comb(new level1Gatherer <edgeMsg_t>, new HelperNode <edgeMsg_t>, true, true)});
        globalLeft.push_back(pipe);
        auto g = a2a.createGroup("W" + std::to_string(i));
        g << pipe;
    }
    a2a.add_firstset(globalLeft, true);
    level0Gatherer <edgeMsg_t> root;
    a2a.add_secondset<ff_node>({&root});
    a2a.createGroup("W0") << root;

#ifdef DISABLE_FF_DISTRIBUTED
    if (a2a.run_and_wait_end() < 0) {
        error("Error while executing: All-to-All\n");
        return -1;
    }
#else
    ff::ff_pipeline pipe;
    pipe.add_stage(&a2a);
    if (pipe.run_and_wait_end() < 0) {
        error("Error while executing: Pipe\n");
        return -1;
    }
#endif

    if (groupName.compare(loggerName) == 0)
        chrono.stop();
    return 0;
}
