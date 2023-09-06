#include "helpers.hpp"

// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat &frame) {
    // processing cv image to adapt to the model input
    cv::Mat img;

    cv::resize(frame, img, cv::Size(1280, 720));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    //std::cout << img.size() << std::endl;
    //std::cout << "rows" << img.rows << std::endl;
    //std::cout << "cols" << img.cols << std::endl;
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, torch::kByte);

    // add batch dimension, from [3,640,640] to [1,3,640,640]
    return imgTensor.toType(torch::kFloat).permute({2, 0, 1}).div(255).unsqueeze(0);
}

// Convert eth image features from MAt to Tensor
torch::Tensor featToTensor(const cv::Mat &feat) {
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

// Convert a video frame form tensor to opencv mat format.permute({1, 0, 2})
// TODO: check this
cv::Mat tensorToFeat(const torch::Tensor &tensor) {
    torch::Tensor buffer = tensor.squeeze(0).permute({1, 2, 0}).toType(torch::kFloat).contiguous();
    return cv::Mat(buffer.size(0), buffer.size(1), CV_32F, buffer.data_ptr<float>());
}

cv::Mat tensorToProjectionMat(const torch::Tensor &tensor) {
    return cv::Mat(tensor.size(1), tensor.size(2), CV_64F, tensor.data_ptr<double>());
}

void show_results(const cv::Mat &frame) {
    cv::imshow("", frame);
    cv::waitKey(1);
}





