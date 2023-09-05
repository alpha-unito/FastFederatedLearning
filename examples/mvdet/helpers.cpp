#include "helpers.hpp"

// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat& frame) {
  // processing cv image to adapt to the model input
  cv::Mat img;
  // cv::resize(frame, img, cv::Size(MOD_W, MOD_H));
  cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);
  torch::Tensor imgTensor =
      torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
  imgTensor = imgTensor.permute({2, 0, 1});
  imgTensor = imgTensor.toType(torch::kFloat);
  imgTensor = imgTensor.div(255);
  imgTensor = imgTensor.unsqueeze(
      0);  // add batch dimension, from [3,640,640] to [1,3,640,640]

  return imgTensor;
}

// Convert the image features from MAt to Tensor
torch::Tensor featToTensor(const cv::Mat& feat) {
  // processing cv image to adapt to the model input

  torch::Tensor imgTensor =
      torch::from_blob(feat.data, {feat.rows, feat.cols, 512}, torch::kByte);
  imgTensor = imgTensor.permute({2, 0, 1});
  imgTensor = imgTensor.toType(torch::kFloat);
  imgTensor = imgTensor.unsqueeze(
      0);  // add batch dimension, from [3,640,640] to [1,3,640,640]

// [1, 512, 120, 360]

  return imgTensor;
}

// Convert a video frame form tensor to opencv mat format
// TODO: check this
// cv::Mat tensorToFeat(const torch::Tensor &tensor) {
//   std::vector< int > sizes;

//   for(auto size : tensor.sizes())
//     sizes.push_back(size);

//   return cv::Mat(sizes,
//                    CV_32F,
//                    tensor.data_ptr<float>());
// }

cv::Mat tensorToFeat(torch::Tensor& tensor) {
    tensor = tensor.squeeze().detach();
    tensor = tensor.permute({1, 2, 0}).contiguous();
    // tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    return cv::Mat(cv::Size(width, height), CV_32F, tensor.data_ptr<float>());
}

cv::Mat tensorToProjectionMat(const torch::Tensor &tensor) {

    return cv::Mat(tensor.size(1),
                   tensor.size(2),
                   CV_64F,
                   tensor.data_ptr<double>());
}

void show_results(const cv::Mat& frame) {
  cv::imshow("", frame);
  cv::waitKey(1);
}





