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

#include <torch/torch.h>
#include <ff/dff.hpp>
#include <iostream>
#include <chrono>
#include <mutex>

#include <dfl/federation.hpp>
#include <dfl/fedavg.hpp>


using namespace ff;


// Define a new Module.
struct Net : torch::nn::Module {
    Net() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

template<typename T>
struct MiNodeAdapter : ff_minode_t<T> {
    T *svc(T *in) { return in; }
};


// ignore for shared memory version
template<typename T>
void serializefreetask(T *o, Net *input) {
    //ff::cout << "serializeFreeTask!" << std::endl;
}

// cereal based serialization do not work
template<class Archive>
void save(Archive &archive, Net const &m) {
    std::ostringstream oss;
    torch::serialize::OutputArchive o;
    m.save(o);
    o.save_to(oss);
    archive(oss.str());
}

template<class Archive>
void load(Archive &archive, Net &m) {
    std::string r;
    archive(r);
    std::istringstream iss(r);
    torch::serialize::InputArchive i;
    i.load_from(iss);
    m.load(i);
}

int main(int argc, char *argv[]) {
    std::chrono::time_point <std::chrono::system_clock> start_time;
    std::chrono::time_point <std::chrono::system_clock> end_time;

    start_time = std::chrono::system_clock::now();

    // required by the distributed version of FF
    DFF_Init(argc, argv);

    int num_workers = 3;                // Number of workers
    int train_batchsize = 64;           // Train batch size
    int test_batchsize = 1000;          // Test batch size
    int train_epochs = 2;               // Number of training epochs at workers in each round
    int rounds = 10;                     // Number of training rounds
    std::string data_path = "../../../data";  // Patch to the MNISt data files (absolute or with respect to build directory)
    int forcecpu = 0;
    int nt = 4; // Number of threads per process

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
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
    std::cout << "Training on " << num_workers << " wokers." << std::endl;

    // Use GPU, if available
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && !forcecpu) {
        //std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        //std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    //torch::set_num_threads(nt);

    torch::cuda::manual_seed_all(42);

    // Get train and test data
    auto train_dataset = torch::data::datasets::MNIST(data_path)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto test_dataset = torch::data::datasets::MNIST(data_path, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    ff_a2a a2a;

    // Create set of workers each with its own model, optimizer and training set
    std::vector < ff_node * > w;
    for (int i = 0; i < num_workers; ++i) {
        auto net = new Net;
        auto optimizer = std::make_shared<torch::optim::SGD>(net->parameters(),
                                                             torch::optim::SGDOptions(0.01).momentum(0.5));
        // HACK: in sampler set total number of nodes to 8 subdivide always the data in 8 partitions
        ff_node *worker = new ff_comb(new MiNodeAdapter<Net>, new Worker(i, net, train_epochs, optimizer,
                                                                         torch::data::make_data_loader(train_dataset,
                                                                                                       torch::data::samplers::DistributedRandomSampler(
                                                                                                               train_dataset.size().value(),
                                                                                                               8/*num_workers*/,
                                                                                                               i % 8,
                                                                                                               false),
                                                                                                       train_batchsize),
                                                                         device));
        w.push_back(worker);
        a2a.createGroup("W" + std::to_string(i)) << worker;
    }

    a2a.add_secondset(w); // M2,M4,M6 All right

    // Create federator with its own model, aggregator and testing set
    auto net = new Net;
    FedAvg aggregator(net);
    ff_node *federator = new ff_comb(new MiNodeAdapter<Net>, new Federator(net, aggregator, num_workers, rounds,
                                                                           torch::data::make_data_loader(test_dataset,
                                                                                                         test_batchsize),
                                                                           device));

    a2a.add_firstset<ff_node>({federator}); // M1,M3,M5 all left

    // distributed-memory Federator group
    a2a.createGroup("Federator") << federator;

    // required for the distributed-version
#ifndef DISABLE_FF_DISTRIBUTED
    ff::ff_pipeline pipe;
    pipe.add_stage(&a2a);
    pipe.wrap_around();  // for distributed memory version
    pipe.run_and_wait_end(); // for distributed memory version
#else
    a2a.wrap_around();
    a2a.run_and_wait_end();
#endif

    end_time = std::chrono::system_clock::now();

    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << "ms" << std::endl;

    return 0;
}
