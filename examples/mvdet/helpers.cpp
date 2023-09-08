#include "helpers.hpp"

// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat &frame) {
    // processing cv image to adapt to the model input
    cv::Mat img;

    cv::resize(frame, img, cv::Size(1280, 720));
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, torch::kByte);
    // add batch dimension, from [3,640,640] to [1,3,640,640]
    return imgTensor.toType(torch::kFloat).permute({2, 0, 1}).div(255.0).unsqueeze(0);
}

// Convert a view tensor to opencv mat format (black and white)
cv::Mat tensorToImg(const torch::Tensor &tensor, int mult) {
    torch::Tensor buffer = tensor.squeeze(0).permute({1, 2, 0}).mul(mult).toType(torch::kByte);
    return cv::Mat(buffer.size(0), buffer.size(1), CV_8UC(1), buffer.data_ptr<uchar>());
}

// Convert eth image features from MAt to Tensor
torch::Tensor featToTensor(const cv::Mat &feat, float max) {
    // processing cv image to adapt to the model input
    torch::Tensor imgTensor = torch::from_blob(feat.data, {feat.rows, feat.cols, feat.channels()}, torch::kByte);
    // add batch dimension, from [3,640,640] to [1,3,640,640]
    // [1, 512, 120, 360]
    return imgTensor.toType(torch::kFloat).permute({2, 0, 1}).mul(max / 255.0).unsqueeze(0);
}

// Convert a video frame form tensor to opencv mat format
// Max and min are necessary for de-normalising the image
cv::Mat tensorToFeat(const torch::Tensor &tensor) {
    float max = tensor.max().item().toFloat();
    float min = tensor.min().item().toFloat();
    torch::Tensor buffer = tensor.squeeze(0).permute({1, 2, 0}).sub(min).mul(255.0 / (max - min)).toType(torch::kByte);
    return cv::Mat(buffer.size(0), buffer.size(1), CV_8UC(buffer.size(2)), buffer.data_ptr<uchar>()).clone();
}

cv::Mat tensorToProjectionMat(const torch::Tensor &tensor) {
    return cv::Mat(tensor.size(1), tensor.size(0), CV_32F, tensor.data_ptr());
}

void show_results(const cv::Mat &frame, const std::string title) {
    cv::imshow(title, frame);
    cv::waitKey(0);
}

void show_results(const torch::Tensor &tensor, const std::string title) {
    float max = tensor.max().item().toFloat();
    std::string max_str = std::to_string(max);
    float min = tensor.min().item().toFloat();
    std::string min_str = std::to_string(min);
    cv::Mat dst = tensorToFeat(tensor);
    show_results(dst, title + " Max: " + max_str + " Min: " + min_str);
}


