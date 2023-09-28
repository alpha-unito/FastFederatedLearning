#include <ff/ff.hpp>
#include <iostream>
#include "C/dfl/traintest.hpp"
#include "C/utils/utils.hpp"

template<typename StateDict, typename Model, typename Aggregator>
class Federator : public ff::ff_monode_t<StateDict> {
private:
    StateDict *state_dict;
    Model *model;
    Aggregator aggregator;
    int num_workers_per_round;
    int rounds;
    int round{0};
    int k{0};
public:
    Federator() = delete;

    Federator(StateDict *state_dict, Model *model, Aggregator agg, int num_workers_per_round, int rounds) :
            state_dict(state_dict),
            model(model),
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
            for (int i = 0; i < num_workers_per_round; ++i)
                this->ff_send_out_to(state_dict, i);
        }

        return this->GO_ON;
    }
};

template<typename Net, typename StateDict, typename Optimizer, typename TrainDataLoader, typename TestDataLoader>
class Worker : public ff::ff_monode_t<StateDict> {
private:
    ssize_t id_;
    StateDict *model;
    Net *net;
    Optimizer optimizer;
    TrainDataLoader train_data_loader;
    TestDataLoader test_data_loader;
    torch::Device device;
    int epoch{0};
    int train_epochs;
public:
    Worker() = delete;

    Worker(ssize_t id, Net *net, StateDict *loc_model, int train_epochs, Optimizer &loc_optimizer,
           TrainDataLoader &&train_data_loader, TestDataLoader &&test_data_loader, torch::Device device = torch::kCPU) :
            id_(id),
            train_data_loader(std::move(train_data_loader)),
            test_data_loader(std::move(test_data_loader)),
            train_epochs(train_epochs),
            device(device),
            net(net),
            model(loc_model),
            optimizer(std::move(loc_optimizer)) { model; }

    StateDict *svc(StateDict *task) {
        copy_model(*net->state_dict(), *task);
#ifndef DISABLE_FF_DISTRIBUTED
        delete task;
#endif

        std::cout << "[" << id_ << "] Starting local testing..." << std::endl;
        test(net, device, *test_data_loader, std::to_string(id_));

        std::cout << "[" << id_ << "] Starting local training..." << std::endl;
        for (int i = 0; i < train_epochs; i++) {
            std::cout << "[" << id_ << "] Epoch " << i << "..." << std::endl;
            train(++epoch, net, device, *train_data_loader, *optimizer, std::to_string(id_));
        }

        return net->state_dict();
    }
};

template<typename Model, typename StateDict, typename Optimizer, typename DataLoaderTrain, typename Aggregator, typename DataLoaderTest>
class Peer : public ff::ff_monode_t<StateDict> {
private:
    ssize_t id_;
    Model *loc_net;
    StateDict *loc_model;
    StateDict *fed_model;
    Optimizer optimizer;
    int epoch{0};
    int train_epochs;
    DataLoaderTrain train_data_loader;
    torch::Device device;
    DataLoaderTest test_data_loader;
    Model *fed_net;
    Aggregator aggregator;
    int num_workers_per_round;
    int rounds;
    int round{0};
    int k{0};

public:
    Peer() = delete;

    Peer(ssize_t id, Model *loc_net, StateDict *loc_model, int train_epochs, Optimizer &loc_optimizer,
         DataLoaderTrain &&train_data_loader, Model *fed_net, StateDict *fed_model, Aggregator agg,
         int num_workers_per_round, int rounds,
         DataLoaderTest &&test_data_loader, torch::Device device = torch::kCPU) :
            id_(id),
            train_data_loader(std::move(train_data_loader)),
            train_epochs(train_epochs),
            device(device),
            loc_net(loc_net),
            loc_model(loc_model),
            fed_model(fed_model),
            fed_net(fed_net),
            optimizer(std::move(loc_optimizer)),
            num_workers_per_round(num_workers_per_round),
            rounds(rounds),
            test_data_loader(std::move(test_data_loader)),
            aggregator(agg) {} // TODO: to_device

    StateDict *svc(StateDict *task) {
        if (task == nullptr)
            k = 0;
        else {
            ++k;
            //printf("[%ld] Received %d-th model\n", id_, k);
            aggregator.update_from(*task);
            delete task;
        }

        if (k >= num_workers_per_round) {
            // We completed one round
            ++round;
            k = 0;

            test(fed_net, device, *test_data_loader, std::to_string(id_));

            if (round >= rounds) {
                printf("\n[%ld] Finished training!\n", id_);
                return this->EOS;
            }

            copy_model(*loc_net->state_dict(), *fed_model);
        }

        if (k == 0) {
            // Start a new round
            //printf("\n[%ld] Starting round %d\n", id_, round);
            aggregator.new_round();
            // Train model with local data
            //std::cout << "Epochs: " << train_epochs << std::endl;
            for (int i = 0; i < train_epochs; i++)
                train(++epoch, loc_net, device, *train_data_loader, *optimizer, std::to_string(this->get_my_id()));

            // ... and add it to the federated model
            aggregator.update_from(*loc_model);
            ++k;

            this->ff_send_out_to(loc_net->state_dict(), id_);
        }

        return this->GO_ON;
    }
};

template<typename Model>
class Distributor : public ff::ff_monode_t<Model> {
private:
    torch::Device device_;
    ssize_t id_;
    ssize_t num_peers_;
public:
    Distributor() = delete;

    Distributor(ssize_t id, ssize_t num_peers, torch::Device device = torch::kCPU) :
            device_(device),
            id_(id),
            num_peers_(num_peers) {}

    Model *svc(Model *task) {
        // Forward model to all other peers
        for (ssize_t i = 0; i < num_peers_; i++) {
            if (i != id_)// Skip ourself
                this->ff_send_out_to(task, i);
        }
        delete task;
        return this->GO_ON;
    }
};
