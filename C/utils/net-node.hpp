#include <iostream>
#include <ff/ff.hpp>

#include "net.hpp"

// struct TensorWrapper {
//     at::Tensor wrapped_data;
//     TensorWrapper() {}

//     TensorWrapper(at::Tensor data) : wrapped_data(data) {}

//     at::Tensor data() const { return wrapped_data; }
// };


class NetNode : public ff::ff_node_t< at::Tensor> {
private:
    Net <torch::jit::Module> *net;
    torch::Device device;

public:
    NetNode() = delete;

    NetNode(std::string &model_path, torch::Device device = torch::kCPU) :
            device(device) {
        net = (new Net<torch::jit::Module>(model_path));
        net->to(device);
    }
    ~NetNode() {
        delete net;
    }

    at::Tensor *svc( at::Tensor *input) {

        at::Tensor* output = new at::Tensor(net->forward(input->to(device)));
        delete input;

        return output;
    }
};