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

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include <nlohmann/json.hpp>
#include <fstream>
#include <list>
#include <algorithm>
#include <filesystem>
#include <chrono>

//using json = nlohmann::json;
using namespace std;

const int IL = 0;  // index of left coordinate in tensor
const int IT = 1;  // index of top coordinate in tensor
const int IR = 2;  // index of right coordinate in tensor
const int IB = 3;  // index of bottom coordinate in tensor

namespace cereal {

    template <class Archive, class T>
    void save(Archive& ar, const cv::Point_<T>& point) {
        ar << point.x << point.y;
    }

    template <class Archive, class T>
    void load(Archive& ar, cv::Point_<T>& point) {
        ar >> point.x >> point.y;
    }
}  // namespace cereal

class BboxInfo {
    public:
        cv::Point p1;
        cv::Point p2;
        BboxInfo() {}

        BboxInfo(const cv::Point& x1, const cv::Point& x2): p1(x1), p2(x2) {}

        BboxInfo(const BboxInfo& b) : p1(b.p1), p2(b.p2) {}
        
        template<class Archive>
	    void serialize(Archive & archive) {
		    archive(p1,p2);
	    }
};



//  // Algorithm man down properties:
extern const float RATIO_THRESH;
//extern std::list<std::string> classes_to_detect;

extern bool isMandown;


inline float left(const torch::Tensor& x) {
  return x[IL].item().toFloat();
}

inline float right(const torch::Tensor& x) {
  return x[IR].item().toFloat();
}

inline float top(const torch::Tensor& x) {
  return x[IT].item().toFloat();
}

inline float bottom(const torch::Tensor& x) {
  return x[IB].item().toFloat();
}

// Computes the areas of overlap between the best current bbox (found in position indexes[0]) and the rest of the bboxes.
// result is a tensor of size indexes.sizes()[0] - 1 containing the computed areas.
// 
// Parameters:
//  - dets is a tensor containing the bboxes to compare with the best current bbox.
//  - indexes is a tensor containing the indexes of the best current bbox.
torch::Tensor compute_overlaps(torch::Tensor dets, torch::Tensor indexes);


// Implements the non_max_suppression operation for a single image.
//
// Original code from: https://github.com/Nebula4869/YOLOv5-LibTorch/blob/master/src/YOLOv5LibTorch.cpp
// Differences:
// - this version is rewritten to be better readable.
// - It removes an unnecessary loop (the outer most loop, did nothing since it iterated only once on our data, the original version
//   was able to process multiple images at once).
//

// Notes:
// -  pred.select(1,4) is the same as pred[:,4], i.e., it takes only the confidence score for all 25200 bounding boxes (bbox in the following).
// - pred.slice(1,5,pred.size()[1]) is the same as pred[:,5:], i.e., it takes only the class scores for each bounding box.
// - the scores variable contains, hence, the (score * class_score) for each bbox and the best class for that bbox
torch::Tensor non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh);

// Converts a video frame into a tensor that can be used as input to the model.
torch::Tensor imgToTensor(const cv::Mat& frame);

void show_results(const cv::Mat &frame, const std::vector<BboxInfo>& bboxes);

std::vector<BboxInfo> compute_bboxes(const cv::Mat &frame, const torch::Tensor& dets);

bool man_down(const BboxInfo bbox);

void process_results(std::vector<BboxInfo>& results);

// Process the given video using the YOLOv5 model.
void process_video(torch::jit::script::Module& model, std::string video_fname, bool show_video);
