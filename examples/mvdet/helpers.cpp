#include "helpers.hpp"

// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat& frame) {
  // processing cv image to adapt to the model input
  cv::Mat img;
  // cv::resize(frame, img, cv::Size(MOD_W, MOD_H));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  torch::Tensor imgTensor =
      torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
  imgTensor = imgTensor.permute({2, 0, 1});
  imgTensor = imgTensor.toType(torch::kFloat);
  imgTensor = imgTensor.div(255);
  imgTensor = imgTensor.unsqueeze(
      0);  // add batch dimension, from [3,640,640] to [1,3,640,640]

  return imgTensor;
}

// Convert a video frame form tensor to opencv mat format
// TODO: check this
cv::Mat tensorToImg(const torch::Tensor &tensor) {

    return cv::Mat(tensor.size(0),
                   tensor.size(1),
                   CV_32F,
                   tensor.data_ptr<float>());
}

cv::Mat tensorToProjectionMat(const torch::Tensor &tensor) {

    return cv::Mat(tensor.size(0),
                   tensor.size(1),
                   CV_32F,
                   tensor.data_ptr<float>());
}

void show_results(const cv::Mat& frame) {
  cv::imshow("", frame);
  cv::waitKey(1);
}





