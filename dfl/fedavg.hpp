#include <torch/torch.h>

template<typename Model>
struct FedAvg {
    FedAvg() = delete;

    FedAvg(Model &model) :
            total_weight{0.0},
            model{model} {
    }

    void new_round() {
        total_weight = 0.0;
    }

    void update_from(Model &src, double weight = 1.0) {
        torch::NoGradGuard guard;
        double new_total_weight = total_weight + weight;

        // Iterate over the model parameters
        assert(model->parameters().size() == src->parameters().size());
        for (int j = 0; j < model->parameters().size(); j++) {
            torch::Tensor p_dst = model->parameters().at(j);
            torch::Tensor p_src = src->parameters().at(j);

            // Update parameters
            // TODO: consider storing the product and avoid one multiplication
            p_dst.data().mul_(total_weight);
            p_dst.data().add_(p_src.data() * weight);
            p_dst.data().div_(new_total_weight);
        }

        // Iterate over the model buffers
        assert(model->buffers().size() == src->buffers().size());
        for (int j = 0; j < model->buffers().size(); j++) {
            torch::Tensor p_dst = model->buffers().at(j);
            torch::Tensor p_src = src->buffers().at(j);

            // Update buffers
            p_dst.copy_(p_src); //HACK
            // p_dst.data().mul_(total_weight);
            // p_dst.data().add_(p_src.data() * weight);
            // p_dst.data().div_(new_total_weight);
        }
        total_weight = new_total_weight;
    }

private:
    Model model;
    double total_weight;
};