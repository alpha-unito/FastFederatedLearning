#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>

// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat &frame);

torch::Tensor featToTensor(const cv::Mat &feat, float max);

// to check
cv::Mat tensorToImg(const torch::Tensor &tensor, int mult);

cv::Mat tensorToFeat(const torch::Tensor &tensor);

cv::Mat tensorToProjectionMat(const torch::Tensor &tensor);

void show_results(const cv::Mat &frame, const std::string title);

void show_results(const torch::Tensor &frame, const std::string title);
