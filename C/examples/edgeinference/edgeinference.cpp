/*
 * FastFlow concurrent network:
 *
 *  -----------------------------------------------------------
 * |  /<------------ a2a(0)----------->/                        |
 * |   -------------------------------                          |
 * |  |                               |                         |
 * |  |  Client ->|                   |                         |
 * |  |           | -> level1Gatherer | ->|                     |
 * |  |  Client ->|                   |   |                     |
 * |  |                               |   |                     |
 * |   -------------------------------    |                     |
 * |       ....                           | --> level0Gatherer  |
 * |                                      |                     |
 * |   -------------------------------    |                     |
 * |  |                               |   |                     |
 * |  |  Client ->|                   |   |                     |
 * |  |           | -> level1Gatherer | ->|                     |
 * |  |  Client ->|                   |                         |
 * |  |                               |                         |
 * |   -------------------------------                          |
 * |  /<------------ a2a(n)----------->/                        |
 * |                                                            |
 *  ------------------------------------------------------------
 * /<-------------------------- mainA2A ----------------------->/
 *
 *
 * distributed version:
 *
 *  each a2a(i) is a group, a2a(i) --> Gi i>0
 *  level0Gatherer: G0
 *
 */


#include <iostream>

#include <ff/dff.hpp>
#include <torch/torch.h>

#include "process-frame.hpp"
#include "C/utils/serialize.hpp"

using namespace ff;

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

struct EdgeNode : ff_monode_t<edgeMsg_t> {
    EdgeNode(std::string workerName, torch::jit::script::Module model, std::string video_fname) : workerName(
            workerName), model(model), video_fname(video_fname) {}

    int svc_init() {
        std::cout << "[" << workerName << "] Starting to process video..." << video_fname << std::endl;
        cap = cv::VideoCapture(video_fname);
        start = std::chrono::steady_clock::now();
        frame_count = 0;
        return 0;
    }

    void svc_end() {
        // 	// Close the file?
        cap.release();
        std::cout << "Video closed, process finished. " << std::endl;
    }

    edgeMsg_t *svc(edgeMsg_t *) {

        while (cap.isOpened()) {
            frame_count++;
            std::cout << "[" << workerName << "] Processing frame..." << std::endl;
            cap.read(frame);
            if (frame.empty()) {
                std::cout << "[" << workerName << "] Read frame failed!" << std::endl;
                break;
            }

            torch::Tensor imgTensor = imgToTensor(frame);
            torch::Tensor preds = model.forward({imgTensor}).toTuple()->elements()[0].toTensor();

            torch::Tensor detections = non_max_suppression(preds, 0.5, 0.3);
            std::vector <BboxInfo> bboxes = compute_bboxes(frame, detections);

            //process_results(bboxes);
            end = std::chrono::steady_clock::now();
            elapsed = end - start;
            uint64_t elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

            edgeMsg_t *task = new edgeMsg_t;
            task->bboxes = bboxes;
            task->frame_n = frame_count;
            //task->nodeTime=end;
            task->str = "Msg";
            ff_send_out(task);
        }
        return EOS;
    }

    const std::string workerName, video_fname;
    torch::jit::script::Module model;
    std::chrono::steady_clock::time_point start, end;
    std::chrono::steady_clock::duration elapsed;
    unsigned long frame_count;
    cv::VideoCapture cap;
    cv::Mat frame;

};

struct level1Gatherer : ff_minode_t<edgeMsg_t> {
    edgeMsg_t *svc(edgeMsg_t *t) {
        //t->str += std::string(" World");
        auto results = t->bboxes;
        for (auto bbox: results) {
            man_down(bbox);
            if (isMandown == true) {
                //std::cout << bbox.p1 << " " << bbox.p2 << " mandown"<<std::endl;
                edgeMsg_t *task = new edgeMsg_t;
                task->bboxes.push_back(bbox);
                task->str = "Msga";
                ff_send_out(task);
                //}

            }

        }
        delete t;
        return GO_ON;
    }
};


struct HelperNode : ff_monode_t<edgeMsg_t> {
    edgeMsg_t *svc(edgeMsg_t *task) { return task; }
};


struct level0Gatherer : ff_minode_t<edgeMsg_t> {
    edgeMsg_t *svc(edgeMsg_t *task) {
        auto results = task->bboxes;
        for (auto bbox: results) {
            //std::cout << "mandown in root "<< bbox.p1 << " " << bbox.p2 << std::endl;
        }
        // std::cerr << "level0Gatherer: from (" << get_channel_id() << ") " << t->str << "frame count " << t->S.f;
        return GO_ON;
    }
};


int main(int argc, char *argv[]) {
    timer chrono = timer("Total execution time");

#ifndef DISABLE_FF_DISTRIBUTED
    DFF_Init(argc, argv);
#endif

    long ntasks = 100;
    size_t nL = 3;
    size_t pL = 1; // 1 client per group
    int forcecpu = 0;
    std::string inmodel = "../../../data/yolov5n.torchscript";
    std::string infile = "../../../data/Ranger_Roll_m.mp4";

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
            std::cout << "Usage: edgeinference[forcecpu=0/1] [groups=3] [clients/group=1] [model_path] [data_path]\n";
            exit(0);
        } else
            forcecpu = atoi(argv[1]);
    }
    if (argc >= 3)
        nL = atoi(argv[2]);
    if (argc >= 4)
        pL = atoi(argv[3]);
    if (argc >= 5)
        inmodel = argv[4];
    if (argc >= 6)
        infile = argv[5];

    torch::DeviceType device_type;
    if (torch::cuda::is_available() && !forcecpu) {
        //std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        //std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    torch::cuda::manual_seed_all(42);

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(inmodel);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n" << std::endl
                  << "Details:" << std::endl
                  << e.what() << std::endl;
        return -1;
    }
    //  TBD one or more file per node
    //if (argc>3) infile = argv[2];
    if (std::filesystem::exists(infile) == false) {
        std::cerr << "Video file cannot be found at path: " << argv[2] << std::endl;
        return -1;
    } //else infile=std::string(argv[2]);

    ff_a2a mainA2A;
    level0Gatherer root;
    std::vector < ff_node * > globalLeft;

    for (size_t i = 0; i < nL; ++i) {
        ff_pipeline *pipe = new ff_pipeline;   // <---- To be removed and automatically added
        ff_a2a *a2a = new ff_a2a;
        pipe->add_stage(a2a, true);
        std::vector < ff_node * > localLeft;
        for (size_t j = 0; j < pL; ++j)
            localLeft.push_back(new EdgeNode("W(" + std::to_string(i) + "," + std::to_string(j) + ")", model, infile));
        a2a->add_firstset(localLeft, 0, true);
        a2a->add_secondset<ff_comb>({new ff_comb(new level1Gatherer, new HelperNode, true, true)});
        globalLeft.push_back(pipe);
        // create here Gi groups for each a2a(i) i>0
        auto g = mainA2A.createGroup("G" + std::to_string(i + 1));
        g << pipe;
    }

    mainA2A.add_firstset(globalLeft, true);
    mainA2A.add_secondset<ff_node>({&root});

    mainA2A.createGroup("G0") << root;

    if (mainA2A.run_and_wait_end() < 0) {
        error("running the main All-to-All\n");
        return -1;
    }

    chrono.stop();
    return 0;
}
