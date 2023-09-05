#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>


// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat& frame);

// to check
cv::Mat tensorToFeat(torch::Tensor& tensor);

cv::Mat tensorToProjectionMat(const torch::Tensor &tensor);

void show_results(const cv::Mat &frame);