/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

#include "process-frame.hpp"

const int MOD_W = 640;  // width of model input tensor
const int MOD_H = 640;  // height of model input tensor

const int ICX = 0;  // index of center x coordinate in tensor
const int ICY = 1;  // index of center y coordinate in tensor
const int IW = 2;   // index of width in tensor
const int IH = 3;   // index of height in tensor

const float RATIO_THRESH = 1.0;
//std::list<std::string> classes_to_detect = {"person", "boat"};

bool isMandown;

torch::Tensor compute_overlaps(torch::Tensor dets, torch::Tensor indexes) {
  const int isizes = indexes.sizes()[0] - 1;
  const torch::Tensor L = indexes[0];  // Largest bbox index

  const float Ll = left(dets[L]);
  const float Lt = top(dets[L]);
  const float Lr = right(dets[L]);
  const float Lb = bottom(dets[L]);

  torch::Tensor widths = torch::empty(isizes);
  torch::Tensor heights = torch::empty(isizes);

  for (size_t i = 0; i < isizes; ++i) {
    auto elem = dets[indexes[i + 1]];
    float l = std::max(Ll, left(elem));
    float t = std::max(Lt, top(elem));
    float r = std::min(Lr, right(elem));
    float b = std::min(Lb, bottom(elem));

    widths[i] = std::max(float(0), r - l);
    heights[i] = std::max(float(0), b - t);
  }

  return widths * heights;
}

// Assumes that tensor contains coordinates expressed as (center_x, center_y, w,
// h) and rewrites it in-place so that they are expressed as (left, top, right,
// bottom)
void xywh_to_xyxy(torch::Tensor& tensor) {
  // Note that in the implementation we use the fact that tensors are reference
  // types, so when we change tensor.select(1, IL) we are actually also changing
  // cx (since IL==ICX).
  torch::Tensor cx = tensor.select(1, ICX), cy = tensor.select(1, ICY),
                w = tensor.select(1, IW), h = tensor.select(1, IH);

  tensor.select(1, IL) = cx - w / 2;
  tensor.select(1, IT) = cy - h / 2;
  tensor.select(1, IR) = cx + w;  // note cx already changed to cx - w/2
  tensor.select(1, IB) = cy + h;
}

// Implements the non_max_suppression operation for a single image.
//
// Original code from:
// https://github.com/Nebula4869/YOLOv5-LibTorch/blob/master/src/YOLOv5LibTorch.cpp
// Differences:
// - this version is rewritten to be better readable.
// - It removes an unnecessary loop (the outer most loop, did nothing since it
// iterated only once on our data, the original version
//   was able to process multiple images at once).
//

// Notes:
// -  pred.select(1,4) is the same as pred[:,4], i.e., it takes only the
// confidence score for all 25200 bounding boxes (bbox in the following).
// - pred.slice(1,5,pred.size()[1]) is the same as pred[:,5:], i.e., it takes
// only the class scores for each bounding box.
// - the scores variable contains, hence, the (score * class_score) for each
// bbox and the best class for that bbox
torch::Tensor non_max_suppression(torch::Tensor preds, float score_thresh = 0.5,
                                  float iou_thresh = 0.5) {
  torch::Tensor output;
  torch::Tensor pred = preds.select(
      0, 0);  // select(0,0) which is equivalent to preds[0] (size: 25200 x 85)

  // Filter by scores
  torch::Tensor scores =
      pred.select(1, 4) *
      std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));

  // Filter by score threshold
  pred = torch::index_select(
      pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
  if (pred.sizes()[0] == 0) return output;

  xywh_to_xyxy(pred);
  // (center_x, center_y, w, h) to (left, top, right, bottom)

  // Computing scores and classes
  // torch::max returns a tuple of (max_value, max_index)
  std::tuple<torch::Tensor, torch::Tensor> max_tuple =
      torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
  pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
  pred.select(1, 5) = std::get<1>(max_tuple);

  torch::Tensor candidates =
      pred.slice(1, 0, 6);  // pred[:,:6], i.e., throws away all class scores

  torch::Tensor keep = torch::empty({candidates.sizes()[0]});
  torch::Tensor areas = (candidates.select(1, 3) - candidates.select(1, 1)) *
                        (candidates.select(1, 2) - candidates.select(1, 0));
  std::tuple<torch::Tensor, torch::Tensor> indexes_tuple =
      torch::sort(candidates.select(1, 4), /*dim:*/ 0, /*descending:*/ 1);
  torch::Tensor indexes = std::get<1>(indexes_tuple);

  int count = 0;
  while (indexes.sizes()[0] > 0) {
    auto best_i = indexes[0].item().toInt();
    keep[count] = best_i;
    count += 1;
    torch::Tensor overlaps = compute_overlaps(candidates, indexes);

    // FIlter by IOUs
    // IOU is evaluated as overlap / (area_best + area_other - overlap), but it
    // is done vectorially on the list of currently selected bboxes.
    torch::Tensor ious =
        overlaps / (areas.select(0, best_i) +
                    torch::index_select(
                        areas, /*dim:*/ 0,
                        /*index*/ indexes.slice(0, 1, indexes.sizes()[0])) -
                    overlaps);
    indexes = torch::index_select(
        indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
  }
  keep = keep.toType(torch::kInt64);
  output = torch::index_select(candidates, 0, keep.slice(0, 0, count));

  return output;
}

// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat& frame) {
  // processing cv image to adapt to the model input
  cv::Mat img;
  cv::resize(frame, img, cv::Size(MOD_W, MOD_H));
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

//mandown
bool man_down(const BboxInfo bbox){
     
      if ((bbox.p2.x - bbox.p1.x)/(bbox.p2.y - bbox.p1.y) > RATIO_THRESH ){
        isMandown = true;
      }
      else{
        isMandown = false;  
      }
      return isMandown;
  
}

void show_results(const cv::Mat& frame, const std::vector<BboxInfo>& bboxes) {
  // LOGIC FOR MANDOWN DETECTION
  //     if (std::find(classes_to_detect.begin(), classes_to_detect.end(),
  //     j.at(to_string(classID))) != classes_to_detect.end()){
  //         if (classID == 0){
  //             //mandown
  //             if ((right - left)/(bottom - top) > RATIO_THRESH ){
  //             cv::rectangle(frame, cv::Point(left, top), cv::Point(right,
  //             bottom), cv::Scalar(0, 0, 255), 2); cv::putText(frame,
  //             to_string(j.at(to_string(classID))) + ": " + cv::format("%.2f",
  //             score) ,
  // 	     	        cv::Point(left, top),
  // 			        cv::FONT_HERSHEY_SIMPLEX, (right - left) / 350,
  // cv::Scalar(0, 0, 255), 2);
  //             } else {
  //                 cv::rectangle(frame, cv::Point(left, top), cv::Point(right,
  //                 bottom), cv::Scalar(0, 255, 0), 2);
  //             }
  //         }
  //     }
  // }

  //show red and green rectangles
  for (auto bbox : bboxes) {
    man_down(bbox);

    if(isMandown == true){
      cv::rectangle(frame, bbox.p1, bbox.p2, cv::Scalar(0, 0, 255), 2);
    }
    else{
      cv::rectangle(frame, bbox.p1, bbox.p2, cv::Scalar(0, 255, 0), 2);
    }
  }
  cv::imshow("", frame);
  cv::waitKey(1);
}

std::vector<BboxInfo> compute_bboxes(const cv::Mat& frame,
                                     const torch::Tensor& dets) {
  // loading classes (json file)
  // std::ifstream in("../classes.json");
  // nlohmann::json j;
  // in >> j;

  std::vector<BboxInfo> bboxes;

  // Compute bounding boxes
  for (size_t i = 0; i < dets.sizes()[0]; ++i) {
    float left = (dets[i][IL].item().toFloat() / MOD_W) * frame.cols;
    float top = (dets[i][IT].item().toFloat() / MOD_H) * frame.rows;
    float right = (dets[i][IR].item().toFloat() / MOD_W) * frame.cols;
    float bottom = (dets[i][IB].item().toFloat() / MOD_H) * frame.rows;
    float score = dets[i][4].item().toFloat();
    int classID = dets[i][5].item().toInt();

    bboxes.push_back(BboxInfo(cv::Point(left, top), cv::Point(right, bottom)));
  }

  return bboxes;
}

void process_results(std::vector<BboxInfo>& results) {
   for(auto bbox:results) {
       std::cout << bbox.p1 << " " << bbox.p2 << std::endl;
   }
}

// Process the given video using the YOLOv5 model.
void process_video(torch::jit::script::Module& model, std::string video_fname,
                   bool show_video) {
  std::cout << "Starting to process video..." << std::endl;
  cv::VideoCapture cap = cv::VideoCapture(video_fname);
  cv::Mat frame;
  auto start = std::chrono::steady_clock::now();
  auto frame_count = 0;

  while (cap.isOpened()) {
    frame_count++;

    std::cout << "Processing frame..." << std::endl;
    cap.read(frame);
    if (frame.empty()) {
      std::cout << "Read frame failed!" << std::endl;
      break;
    }

    torch::Tensor imgTensor = imgToTensor(frame);

    torch::Tensor preds =
        model.forward({imgTensor}).toTuple()->elements()[0].toTensor();
    // Output format: [1, 25200, 85], i.e., [batch_size, B, 5 + C] where B is
    // the number of bounding boxes predictedc by the model, C is the number of
    // classes. The first 5 values in the last dimension are (center_x,
    // center_y, w, h, score)

    torch::Tensor detections = non_max_suppression(preds, 0.5, 0.3);
    std::vector<BboxInfo> bboxes = compute_bboxes(frame, detections);

    process_results(bboxes);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "seconds: " << elapsed_seconds.count() << std::endl
              << ((float)frame_count) / elapsed_seconds.count()
              << " frames per second." << std::endl;

    if (show_video) {
      show_results(frame, bboxes);
    }
  }
  cap.release();
  std::cout << "Video closed, process finished. " << std::endl;

}