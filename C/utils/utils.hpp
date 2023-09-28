//
// Created by gmittone on 6/21/23.
//

#ifndef FASTFEDERATEDLEARNING_UTILS_HPP
#define FASTFEDERATEDLEARNING_UTILS_HPP

#include <chrono>

struct timer {
private:
    std::string label;
    std::chrono::time_point <std::chrono::system_clock> start_time;
    std::chrono::time_point <std::chrono::system_clock> end_time;
public:

    timer(std::string label, bool start_timer = true) : label(label) { if (start_timer) start(); }

    void start() {
        start_time = std::chrono::system_clock::now();
    }

    void stop() {
        end_time = std::chrono::system_clock::now();
        std::cout << "[" << label << "]: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms"
                  << std::endl;
    }
};

template<typename Model>
void copy_model(Model &dst, Model &src) {
    torch::NoGradGuard guard;

    // Iterate over the model parameters
    assert(dst.parameters().size() == src.parameters().size());
    for (int j = 0; j < dst.parameters().size(); j++) {
        torch::Tensor p_dst = dst.parameters().at(j);
        torch::Tensor p_src = src.parameters().at(j);

        // Copy model parameters
        p_dst.copy_(p_src);
    }
    // Iterate over the model parameters
    assert(dst.buffers().size() == src.buffers().size());
    for (int j = 0; j < dst.buffers().size(); j++) {
        torch::Tensor p_dst = dst.buffers().at(j);
        torch::Tensor p_src = src.buffers().at(j);

        // Copy model parameters
        p_dst.copy_(p_src);
    }
}

template<typename Model, typename Params>
void copy_to_model(Model &dst, Params &src) {
    torch::NoGradGuard guard;

    assert(dst.parameters().size() == src.parameters().size());
    for (int j = 0; j < dst.parameters().size(); j++) {
        torch::Tensor p_dst = dst.parameters().at(j);
        torch::Tensor p_src = src.parameters().at(j);
        p_dst.copy_(p_src);
    }

    assert(dst.buffers().size() == src.buffers().size());
    for (int j = 0; j < dst.buffers().size(); j++) {
        torch::Tensor p_dst = dst.buffers().at(j);
        torch::Tensor p_src = src.buffers().at(j);
        p_dst.copy_(p_src);
    }
}

#endif //FASTFEDERATEDLEARNING_UTILS_HPP
