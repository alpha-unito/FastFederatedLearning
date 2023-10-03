/* 
 * FastFlow concurrent network:
 *
 * 
 *
 *       2-stages pipeline
 *               
 *          ------------------------------------
 *         |                                    |
 *         |                     ----> W (G1)-->| 
 *         v                    |               | 
 *        Fed (M1) ----------->  ----> W (G2)-->| 
 *         ^                    |               |
 *         |                     ----> W (G3)-->|
 *         |                                    |
 *          ------------------------------------
 *
 *         |<---------  pipe ------------------>|
 *
 *
 */

#include <mutex>
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
    std::string loggerName = "W0";

#ifndef DISABLE_FF_DISTRIBUTED
    for (int i = 0; i < argc; i++)
        if (strstr(argv[i], "--DFF_GName") != NULL) {
            char *equalPosition = strchr(argv[i], '=');
            groupName = std::string(++equalPosition);
            break;
        }
    if (DFF_Init(argc, argv) < 0) {
        error("Error while executing: DFF_Init\n");
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
    char *data_path;            // Patch to the MNISt data files (absolute or with respect to build directory)
    char *inmodel;              // Path to a TorchScript representation of a DNN

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
            if (groupName.compare(loggerName) == 0)
                std::cout << "Usage: masterworker [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path]\n";
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
    //.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    auto test_dataset = torch::data::datasets::MNIST(data_path, torch::data::datasets::MNIST::Mode::kTest).map(
            torch::data::transforms::Stack<>());
    //.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))

    if (groupName.compare(loggerName) == 0)
        std::cout << "Worker creation..." << std::endl;
    ff_a2a a2a;
    std::vector < ff_node * > w;
    for (int i = 1; i <= num_workers; ++i) {
        Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
        auto optimizer = std::make_shared<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001));

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
    }
    a2a.add_secondset(w);

    if (groupName.compare(loggerName) == 0)
        std::cout << "Aggreator creation..." << std::endl;
    Net <torch::jit::Module> *net = new Net<torch::jit::Module>(inmodel);
    FedAvg <StateDict> aggregator(*net->state_dict());
    ff_node *federator = new ff_comb(new MiNodeAdapter<StateDict>,
                                     new Federator(net->state_dict(), net, aggregator, num_workers, rounds));
    a2a.add_firstset<ff_node>({federator});
    a2a.createGroup("W0") << federator;

#ifdef DISABLE_FF_DISTRIBUTED
    a2a.wrap_around();
    if (a2a.run_and_wait_end() < 0) {
        error("Error while executing: All-to-All\n");
        return -1;
    }
#else
    ff::ff_pipeline pipe;
    pipe.add_stage(&a2a);
    pipe.wrap_around();
    if (pipe.run_and_wait_end() < 0) {
        error("Error while executing: Pipe\n");
        return -1;
    }
#endif

    if (groupName.compare(loggerName) == 0)
        chrono.stop();
    return 0;
}
