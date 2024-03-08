//
// Created by gmittone on 6/21/23.
//

#ifndef FASTFEDERATEDLEARNING_NET_HPP
#define FASTFEDERATEDLEARNING_NET_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include "utils.hpp"

struct StateDict {
    std::vector <at::Tensor> parameters_data;
    std::vector <at::Tensor> buffers_data;

    StateDict() {}

    StateDict(std::vector <at::Tensor> parameters, std::vector <at::Tensor> buffers) : parameters_data(parameters),
                                                                                       buffers_data(buffers) {}


    std::vector <at::Tensor> parameters() const { return parameters_data; }

    std::vector <at::Tensor> buffers() const { return buffers_data; }
};

template<typename Module>
class Net : Module {
private:
    Module module;
    StateDict *state_dict_data;

public: // TODO: add clone method/constructor
    Net() : module(NULL) {} //TODO: rendere possibile la creazione di una rete anche non da torchscript
    Net(std::string &model_path) : state_dict_data(nullptr) {
        try {
            module = torch::jit::load(model_path.c_str());
        } catch (const c10::Error &e) {
            std::cerr << "Error loading the model\n" << std::endl
                      << "Details:" << std::endl
                      << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    Net(const char *model_path) : state_dict_data(nullptr) {
        try {
            module = torch::jit::load(model_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading the model\n" << std::endl
                      << "Details:" << std::endl
                      << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // ~Net() = delete;

    torch::Tensor forward(torch::Tensor x) {
        return module.forward({x}).toTensor();
    }

    torch::IValue forward_val(torch::Tensor x) {
        return module.forward({x});
    }

    void train(bool on = true) {
        module.train(on);
    }

    void eval() {
        module.eval();
    }

    Net *clone() {
        Net<Module> *r = new Net<Module>();
        r->module = module.deepcopy();
        return r;
    }

    std::vector <at::Tensor> parameters(bool recurse = true) const {
        std::vector <at::Tensor> parameters;
        for (const torch::Tensor &params: module.parameters())
            parameters.push_back(params);
        return parameters;
    }

    std::vector <at::Tensor> buffers(bool recurse = true) const {
        std::vector <at::Tensor> buffers;
        for (const torch::Tensor &buffer: module.buffers())
            buffers.push_back(buffer);
        return buffers;
    }

    StateDict *state_dict() {
        state_dict_data = new StateDict(parameters(), buffers());
        return state_dict_data;
    }

    void save(std::ostream &out) const {
        module.save(out);
    }

    void to(c10::Device device, bool non_blocking = false) {
        module.to(device, non_blocking);
    }
};

#endif //FASTFEDERATEDLEARNING_NET_HPP
