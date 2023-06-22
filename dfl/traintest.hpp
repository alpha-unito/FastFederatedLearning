#include <torch/torch.h>
#include <string.h>


namespace F = torch::nn::functional;

template<typename DataLoader, typename Net>
void train(size_t epoch, Net &model, torch::Device device, DataLoader &data_loader, torch::optim::Optimizer &optimizer,
           std::string prefix = "", int log_interval = 100) {
    // Train model
    model->train();

    // Iterate over training data in batches
    size_t batch_idx = 0;
    int64_t correct = 0;
    int64_t num_samples = 0;
    float running_loss = 0;
    for (auto &batch: data_loader) {
        // Reset gradients.
        optimizer.zero_grad();
        // Execute the model on the input data.
        //std::vector <torch::jit::IValue> inputs;
        //inputs.push_back(batch.data.to(device));
        torch::Tensor output = model->forward(batch.data.to(device));
        //std::vector<torch::jit::IValue> outputs;
        //outputs.push_back(output);
        // Compute a loss value to judge the prediction of our model.
        torch::Tensor loss = F::cross_entropy(output, batch.target.to(device));
        // Compute gradients of the loss w.r.t. the parameters of our model.

        auto pred = output.argmax(1);
        correct += pred.eq(batch.target.to(device)).sum().template item<int64_t>();
        num_samples += batch.data.size(0);
        running_loss += loss.item<float>() * batch.data.size(0);

        loss.backward();
        // Update the parameters based on the calculated gradients.
        optimizer.step();
        // Print progress
        if (batch_idx++ % log_interval == 0) {
            //std::printf("\rTrain %s Epoch: %ld %5llu Loss: %.4f",
            //prefix.c_str(), epoch, batch_idx * batch.data.size(0), loss.item<float>());
        }
    }
    float train_accuracy = static_cast<float>(correct) / num_samples;
    float train_loss = running_loss / num_samples;

    //std::cout << std::setprecision(6) << " Train Loss: " << train_loss
    //                << " Train Acc: " << train_accuracy
    //                << "Trained on " << num_samples << " samples";
    //std::printf("\n");
}

template<typename DataLoader, typename Net>
void test(Net &model, torch::Device device, DataLoader &data_loader, std::string prefix = "0") {
    // Evaluate model
    model->eval();

    torch::NoGradGuard guard;

    // Iterate over test data in batches
    double test_loss = 0;
    size_t dataset_size = 0;
    int32_t correct = 0;
    for (const auto &batch: data_loader) {
        // First dimension is number of examples
        dataset_size += batch.target.size(0);
        auto targets = batch.target.to(device);
        // Execute the model on the input data.
        std::vector <torch::jit::IValue> inputs;
        inputs.push_back(batch.data.to(device));
        torch::Tensor output = model->forward(batch.data.to(device));
        //torch::Tensor output = model.forward(batch.data.to(device));
        // Accumulate loss over all data batches
        torch::Tensor loss_tensor = F::cross_entropy(output, targets,
                                                     F::CrossEntropyFuncOptions().reduction(torch::kMean));
        test_loss += loss_tensor.item<float>() * batch.data.size(0);
        // Count correctly predicted class
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }
    // Print stats
    test_loss = test_loss / dataset_size;
    if (prefix.compare("0") == 0)
        std::printf("\n%s Test set: Average loss: %.4f | Accuracy: %.3f\n",
                    prefix.c_str(), test_loss, static_cast<double>(correct) / dataset_size);
}
