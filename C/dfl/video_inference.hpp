//
// Created by gmittone on 10/3/23.
//

#ifndef FASTFEDERATEDLEARNING_VIDEO_INFERENCE_HPP
#define FASTFEDERATEDLEARNING_VIDEO_INFERENCE_HPP

#include "C/utils/process-frame.hpp"

struct edgeMsg_t {
    edgeMsg_t() {}

    edgeMsg_t(edgeMsg_t *t) {
        str = std::string(t->str);
        frame_n = t->frame_n;
        //nodeTime = t->nodeTime;
        nodeFrameRate = t->nodeFrameRate;
        bboxes = t->bboxes;
    }

    std::string str;
    unsigned long frame_n;
    // std::chrono::steady_clock::time_point nodeTime;
    float nodeFrameRate;
    std::vector <BboxInfo> bboxes;

    template<class Archive>
    void serialize(Archive &archive) {
        archive(str, frame_n, nodeFrameRate, bboxes);
    }

};

template<typename Model, typename Message>
struct EdgeNode : public ff::ff_monode_t<Message> {
private:
    std::string workerName;
    char *data_path;
    Model model;
    std::chrono::steady_clock::time_point start, end;
    std::chrono::steady_clock::duration elapsed;
    unsigned long frame_count;
    cv::VideoCapture cap;
    cv::Mat frame;
public:
    EdgeNode() = delete;

    EdgeNode(std::string workerName, Model model, char *data_path) :
            workerName(workerName), model(model), data_path(data_path) {}

    int svc_init() {
        std::cout << "[" << workerName << "] Starting to process video..." << data_path << std::endl;
        cap = cv::VideoCapture(data_path);
        start = std::chrono::steady_clock::now();
        frame_count = 0;
        return 0;
    }

    Message *svc(Message *) {
        while (cap.isOpened()) {
            frame_count++;
            std::cout << "[" << workerName << "] Processing frame..." << std::endl;
            cap.read(frame);
            if (frame.empty()) {
                std::cout << "[" << workerName << "] Read frame failed!" << std::endl;
                break;
            }

            torch::Tensor imgTensor = imgToTensor(frame);
            torch::Tensor preds = model->forward_val({imgTensor}).toTuple()->elements()[0].toTensor();
            torch::Tensor detections = non_max_suppression(preds, 0.5, 0.3);
            std::vector <BboxInfo> bboxes = compute_bboxes(frame, detections);

            //process_results(bboxes);
            end = std::chrono::steady_clock::now();
            elapsed = end - start;
            uint64_t elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

            Message *task = new Message;
            task->bboxes = bboxes;
            task->frame_n = frame_count;
            //task->nodeTime=end;
            task->str = "Msg";
            this->ff_send_out(task);
        }
        return this->EOS;
    }

    void svc_end() {
        cap.release();
        std::cout << "Video closed, process finished." << std::endl;
    }
};

template<typename Message>
struct level1Gatherer : public ff::ff_minode_t<Message> {
    Message *svc(Message *t) {
        //t->str += std::string(" World");
        auto results = t->bboxes;
        for (auto bbox: results) {
            man_down(bbox);
            if (isMandown == true) {
                //std::cout << bbox.p1 << " " << bbox.p2 << " mandown"<<std::endl;
                Message *task = new Message;
                task->bboxes.push_back(bbox);
                task->str = "Msga";
                this->ff_send_out(task);
                //}

            }

        }
        delete t;
        return this->GO_ON;
    }
};

template<typename Message>
struct HelperNode : public ff::ff_monode_t<Message> {
    Message *svc(Message *task) { return task; }
};

template<typename Message>
struct level0Gatherer : public ff::ff_minode_t<Message> {
    Message *svc(Message *task) {
        auto results = task->bboxes;
        for (auto bbox: results) {
            //std::cout << "mandown in root "<< bbox.p1 << " " << bbox.p2 << std::endl;
        }
        // std::cerr << "level0Gatherer: from (" << get_channel_id() << ") " << t->str << "frame count " << t->S.f;
        return this->GO_ON;
    }
};

#endif //FASTFEDERATEDLEARNING_VIDEO_INFERENCE_HPP
