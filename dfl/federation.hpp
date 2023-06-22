#include <dfl/traintest.hpp>
#include <ff/ff.hpp>
#include <utils/utils.hpp>
#include <iostream>

template<typename StateDict, typename Model, typename Aggregator, typename DataLoader>
class Federator : public ff::ff_monode_t<StateDict> {
private:
    StateDict *state_dict;
    Model *model;
    DataLoader test_data_loader;
    Aggregator aggregator;
    torch::Device device;
    int num_workers_per_round;
    int rounds;
    int round{0};
    int k{0};
public:
    Federator() = delete;

    Federator(StateDict *state_dict, Model* model, Aggregator agg, int num_workers_per_round, int rounds, DataLoader &&test_data_loader,
              torch::Device device = torch::kCPU) :
            state_dict(state_dict),
            model(model),
            test_data_loader(std::move(test_data_loader)),
            device(device),
            num_workers_per_round(num_workers_per_round),
            rounds(rounds),
            aggregator(agg) { model; }

    StateDict *svc(StateDict *task) {
        if (task >= this->GO_OUT)
            return this->GO_ON;

        if (task == nullptr)
            k = 0;
        else {
            ++k;
            aggregator.update_from(*task);
            delete task;
        }

        if (k >= num_workers_per_round) {
            // We completed one round
            ++round;
            k = 0;

            // Test the model
            copy_to_model(model, state_dict);
            test(model, device, *test_data_loader);

            // Check if we are done
            if (round >= rounds) {
                printf("Finished training!\n");
                return this->EOS;
            }
        }

        if (k == 0) {
            // Start a new round
            printf("Starting round %d\n", round);
            aggregator.new_round();
            for (int i = 0; i < num_workers_per_round; ++i) {
                StateDict *send_data = new StateDict(state_dict->parameters(), state_dict->buffers());
                this->ff_send_out_to(send_data, i);
            }
        }

        return this->GO_ON;
    }
};

template<typename Net, typename StateDict, typename Optimizer, typename DataLoader>
class Worker : public ff::ff_monode_t<StateDict> {
public:
    Worker() = delete;

    Worker(ssize_t id, Net *net, StateDict *loc_model, int train_epochs, Optimizer &loc_optimizer,
           DataLoader &&train_data_loader,
           torch::Device device = torch::kCPU) :
            id_(id),
            train_data_loader(std::move(train_data_loader)),
            train_epochs(train_epochs),
            device(device),
            net(net),
            model(loc_model),
            optimizer(std::move(loc_optimizer)) { model; }

    StateDict *svc(StateDict *task) {
        copy_to_model(net, task);
        delete task;

        std::cout << "Epochs: " << train_epochs << std::endl;
        for (int i = 0; i < train_epochs; i++)
            train(++epoch, net, device, *train_data_loader, *optimizer, std::to_string(this->get_my_id()));

        StateDict *send_data = new StateDict(net->parameters(), net->buffers());
        return send_data;
    }

private:
    ssize_t id_;
    StateDict *model;
    Net *net;
    Optimizer optimizer;
    DataLoader train_data_loader;
    torch::Device device;
    int epoch{0};
    int train_epochs;
};

template<typename Model, typename Optimizer, typename DataLoaderTrain, typename Aggregator, typename DataLoaderTest>
class Peer : public ff::ff_monode_t<Model> {
public:
    Peer() = delete;

    Peer(ssize_t id, Model *loc_model, int train_epochs, Optimizer &loc_optimizer, DataLoaderTrain &&train_data_loader,
         Model *fed_model, Aggregator agg, int num_workers_per_round, int rounds, DataLoaderTest &&test_data_loader,
         torch::Device device = torch::kCPU) :
            id_(id),
            train_data_loader(std::move(train_data_loader)),
            train_epochs(train_epochs),
            device(device),
            model(loc_model),
            optimizer(std::move(loc_optimizer)),
            num_workers_per_round(num_workers_per_round),
            rounds(rounds),
            test_data_loader(std::move(test_data_loader)),
            aggregator(agg),
            fed_model(fed_model) {
        model->to(device);
        fed_model->to(device);
    }

    Model *svc(Model *task) {
        if (task == nullptr) {
            k = 0;
        } else {
            ++k;
            //printf("[%ld] Received %d-th model\n", id_, k);
            aggregator.update_from(task);
            delete task;
        }

        if (k >= num_workers_per_round) {
            // We completed one round
            ++round;
            k = 0;

            // Test the model
            test(fed_model, device, *test_data_loader, std::to_string(id_));

            // Check if we are done
            if (round >= rounds) {
                printf("\n[%ld] Finished training!\n", id_);
                return this->EOS;
            }

            // Update local model
            copy_model(model, fed_model);
        }

        if (k == 0) {
            // Start a new round
            //printf("\n[%ld] Starting round %d\n", id_, round);
            aggregator.new_round();
            // Train model with local data
            //std::cout << "Epochs: " << train_epochs << std::endl;
            for (int i = 0; i < train_epochs; i++) {
                train(++epoch, model, device, *train_data_loader, *optimizer, std::to_string(this->get_my_id()));
            }

            // ... and add it to the federated model
            aggregator.update_from(model);
            ++k;

            // Send out the local model to the distributor
            Model *send_model = new Model();
            copy_model(send_model, model);

            this->ff_send_out_to(send_model, id_);
        }

        return this->GO_ON;
    }

private:
    ssize_t id_;
    Model *model;
    Optimizer optimizer;
    int epoch{0};
    int train_epochs;
    DataLoaderTrain train_data_loader;
    torch::Device device;
    DataLoaderTest test_data_loader;
    Model *fed_model;
    Aggregator aggregator;
    int num_workers_per_round;
    int rounds;
    int round{0};
    int k{0};
};

template<typename Model>
class Distributor : public ff::ff_monode_t<Model> {
public:
    Distributor() = delete;

    Distributor(ssize_t id, ssize_t num_peers, torch::Device device = torch::kCPU) : device_(device) {
        id_ = id;
        num_peers_ = num_peers;
    }

    Model *svc(Model *task) {
        // Forward model to all other peers
        for (ssize_t i = 0; i < num_peers_; i++) {
            if (i != id_) { // Skip ourself
                Model *model = new Model();
                model->to(device_);
                copy_model(model, task);

                this->ff_send_out_to(model, i);
            }
        }
        delete task;
        return this->GO_ON;
    }

private:
    torch::Device device_;
    ssize_t id_;
    ssize_t num_peers_;
};
