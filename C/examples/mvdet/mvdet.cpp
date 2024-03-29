#include <filesystem>
#include <cstdint>

#include <ff/dff.hpp>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "C/utils/net.hpp"
#include "C/utils/utils.hpp"

#include "helpers.hpp"
#include "data_stucts.hpp"

// TODO: refactor code

using namespace ff;

template<typename T>
void serializefreetask(T *o, Frame *input) {}

struct Source : ff_node_t<int> {
private:
    int round = -1;
    const int n_tot_cameras;
public:
    Source() = delete;

    Source(const int n_tot_cameras) : n_tot_cameras{n_tot_cameras} {}

    int *svc(int *i) {
        round++;

        // Send wakeup to each camera to process next frame
        std::cout << "Source starting round " << round << std::endl;
        for (int j = 0; j < n_tot_cameras; j++)
            ff_send_out(new int(round));

        if (i < FF_TAG_MIN) delete i;
        return this->GO_ON;
    }
};

struct CameraNode : ff_monode_t<int, Frame> {
private:
    int out_node;
    int lid;
    int32_t max_round;
    std::string camera_id;
    std::string video_path;
    std::string projection_matrix_path;
    std::string base_model_path;
    std::string image_classifier_path;
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Mat perspective_matrix;
    Net <torch::jit::Module> *base_model;
    Net <torch::jit::Module> *img_classifier;
    Frame *fr;
public:
    CameraNode() = delete;

    CameraNode(std::string camera_id, std::string vp, std::string pm, std::string bm,
               std::string cm, int lid, int out_node, int32_t max_round) : camera_id{camera_id}, video_path{vp},
                                                                           projection_matrix_path{pm},
                                                                           base_model_path{bm},
                                                                           image_classifier_path{cm},
                                                                           out_node{out_node}, lid{lid},
                                                                           max_round{max_round} {}

    int svc_init() {
        // perspective matrix [3x3]
        std::cout << "[ Camera " << camera_id << " ] Loading projection matrix: " << projection_matrix_path
                  << std::endl;
        torch::jit::script::Module container = torch::jit::load(projection_matrix_path);
        torch::Tensor pm = container.attr("data").toTensor();
        perspective_matrix = tensorToProjectionMat(pm).clone();

        // Model loading
        std::cout << "[ Camera " << camera_id << " ] Loading base model: " << base_model_path << std::endl;
        base_model = new Net<torch::jit::Module>(base_model_path);
        // std::cout << "[ Camera " << camera_id << " ] Loading image classifier: " << image_classifier_path << std::endl;
        // img_classifier = new Net<torch::jit::Module>(image_classifier_path);

        // Opening video
        std::cout << "[ Camera " << camera_id << " ] Starting to process video..." << video_path << std::endl;
        cap = cv::VideoCapture(video_path);

        // Check if video was found
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera" << std::endl;
            return -1;
        }
        return 0;
    }

    Frame *svc(int *i) {
        // Read next frame
        std::cout << "[ Camera " << camera_id << " ] Reading frame " << *i << std::endl;
        cap.read(frame);

        if (frame.empty() || (max_round >= 0 && *i > max_round)) {
            std::cout << camera_id << " finished video." << std::endl;
            delete i;
            return this->EOS;
        } else {
            // Process frame using base model
            // show_results(frame, "frame");
            torch::Tensor imgTensor = imgToTensor(frame);
            // show_results(imgTensor, "tensor");
            imgTensor[0][0] = imgTensor[0][0].sub(0.485).div(0.229);
            imgTensor[0][1] = imgTensor[0][1].sub(0.456).div(0.224);
            imgTensor[0][2] = imgTensor[0][2].sub(0.406).div(0.225);
            // show_results(imgTensor, "normalised tensor");
            torch::Tensor img_feature = base_model->forward({imgTensor});

            // Upscaling
            //torch::Tensor img_feature_upscaled = torch::nn::functional::interpolate(
            //        img_feature,
            //        torch::nn::functional::InterpolateFuncOptions()
            //                .mode(torch::kBilinear)
            //                .size(std::vector<int64_t>({270, 480}))
            //);
            // Image classifier
            //torch::Tensor img_res = img_classifier->forward({img_feature_upscaled});

            // Create Frame with buffer for output data
            torch::Tensor max = torch::max(img_feature);
            float max_value = max.item<float>();
            fr = new Frame(out_node, lid, *i, max_value);

            // Warp perspective
            cv::Mat img_feature_mat = tensorToImg(img_feature);
            cv::warpPerspective(img_feature_mat, fr->frame, perspective_matrix, {360, 120});

            // Send it out

            std::cout << "[ Camera " << camera_id << " ] Sending [ " << fr->frame.rows << " x " << fr->frame.cols
                      << " x " << fr->frame.channels() << " ] ( " << fr->imgSizeInKBytes() << " KB )" << std::endl;
            this->ff_send_out_to(fr, out_node);
            delete i;
            return this->GO_ON;
        }
    }

    void svc_end() {
        cap.release();
        std::cout << "[ Camera " << camera_id << " ] Video closed, process finished." << std::endl;
    }
};

struct AggregatorNode : ff_minode_t<Frame, Frame> {
private:
    int counter = 0;
    int n_cameras;
    std::string id;
    std::vector <torch::Tensor> buffer;
    torch::TensorList tensors;
    std::string map_classifier_path;
    std::string coord_mat_path;
    Net <torch::jit::Module> *map_classifier;
public:
    AggregatorNode() = delete;

    // tensors(nullptr, n_cameras)
    AggregatorNode(std::string id, std::string mm, std::string cm, int n_cameras) : id{id}, n_cameras{n_cameras},
                                                                                    buffer(n_cameras + 1),
                                                                                    map_classifier_path{mm},
                                                                                    coord_mat_path{cm} {}

    int svc_init() {
        std::cout << "[ Aggregator " << id << " ] Loading map classifier: " << map_classifier_path << std::endl;
        map_classifier = new Net<torch::jit::Module>(map_classifier_path);

        std::cout << "[ Aggregator " << id << " ] Loading coord matrix: " << coord_mat_path << std::endl;
        torch::jit::script::Module container = torch::jit::load(coord_mat_path);
        buffer[n_cameras] = container.attr("data").toTensor(); // last element is always coordinates matrix

        return 0;
    }

    Frame *svc(Frame *f) {
        std::cout << "[ Aggregator " << id << " ] Received square " << f->id_square << " camera " << f->id_camera
                  << " frame " << f->id_frame
                  << std::endl;
        assert(f->id_camera >= 0 && f->id_camera < n_cameras);

        // Save frame as tensor in buffer
        buffer[f->id_camera] = featToTensor(f->frame, f->max);
        counter++;

        // Buffer full? Process it
        if (counter >= n_cameras) {

            // Concatenate list of tensors into one big tensor
            torch::Tensor world_features_cat = torch::cat(buffer, 1);

            // Feed tensor into model
            torch::Tensor view = map_classifier->forward(world_features_cat);
            float max = view.max().item().toFloat();

            //torch::Tensor norm = view.sub(min).div(max.sub(min));
            //show_results(view, "aggregator result");


            //torch::Tensor img_feature_upscaled = torch::nn::functional::interpolate(
            //        view,
            //        torch::nn::functional::InterpolateFuncOptions()
            //                .mode(torch::kBilinear)
            //                .size(std::vector<int64_t>({120, 360}))
            //);
            //torch::Tensor buf = norm.squeeze(0).mul(255.0);
            //std::cout << buf.sizes() << std::endl;

            //buf = torch::norm(buf, 0);
            //std::cout << buf.sizes() << std::endl;

            //buf = buf.toType(torch::kByte);
            //std::cout << buf.sizes() << std::endl;

            //cv::Mat robe = cv::Mat(120, 360, CV_8UC1, buf.data_ptr<uchar>());
            //std::cout << robe.size() << std::endl;
            //show_results(robe, "aggregator result");

            Frame *fr = new Frame(f->id_square, -1, f->id_frame, max);

            fr->frame = tensorToImg(view);
            //show_results(fr->frame, "aggregator result");

            // Send out result
            std::cout << "[ Aggregator " << id << " ] Sending [ " << fr->frame.rows << " x " << fr->frame.cols << " x "
                      << fr->frame.channels() << " ] ( " << fr->imgSizeInKBytes() << " KB )" << std::endl;
            ff_send_out(fr);

            // Next round
            counter = 0;
        }
        delete f;
        return GO_ON;
    }
};

struct ControlRoom : ff_node_t<Frame, int> {
private:
    uint64_t counter = 0;
    uint64_t tot_counter = 0;
    int n_aggregators;
    std::vector<std::chrono::steady_clock::time_point *> starts;
    std::vector <std::chrono::steady_clock::time_point> ends;
public:
    ControlRoom() = delete;

    ControlRoom(int n_aggregators) : n_aggregators{n_aggregators}, starts(n_aggregators, nullptr),
                                     ends(n_aggregators) {}

    int *svc(Frame *f) {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        if (starts[f->id_square] == nullptr) {
            starts[f->id_square] = new std::chrono::steady_clock::time_point(now);
            std::cout << "[ Control Room ] Received first view " << f->id_frame << " from square " << f->id_square
                      << std::endl;
        } else {
            double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - ends[f->id_square]).count();
            std::cout << "[ Control Room ] Received new view " << f->id_frame << " from square " << f->id_square
                      << " (processing time " << elapsed_ms / 1000.0 << " s)" << std::endl;
        }
        ends[f->id_square] = now;

        tot_counter++;

        // show_results(f->frame, "control room result");
        // std::cout << f->frame.rows << "x" << f->frame.col÷s << " channels: " << f->frame.channels() << std::endl;

        delete f;
        counter++;

        // Received all aggragted results? Start new round
        if (counter >= n_aggregators) {
            counter = 0;
            ff_send_out(new int(0));
        }

        return GO_ON;
    }

    void svc_end() {
        if (tot_counter >= 2) {
            std::chrono::steady_clock::time_point start;
            std::chrono::steady_clock::time_point end;

            for (int i = 0; i < n_aggregators; i++) {
                if (starts[i] != nullptr) {
                    start = *(starts[i]);
                    end = ends[i];
                    break; //TODO better way
                }
            }

            for (int i = 0; i < n_aggregators; i++) {
                if (starts[i] != nullptr && *(starts[i]) < start) start = *(starts[i]);
                if (ends[i] > end) end = ends[i];
            }

            double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            uint64_t processed_views = tot_counter - 1;

            std::cout << "[ Control Room ] Processed " << processed_views << " views in " << elapsed_ms / 1000.0
                      << " s ( " << elapsed_ms / 1000.0 / processed_views << " s/view)" << std::endl;
        }
    }
};

int main(int argc, char *argv[]) {
    timer chrono = timer("Total execution time");

    std::string groupName = "S0";
    const std::string loggerName = "S0"; // TODO: il nome del logger non è standard

#ifndef DISABLE_FF_DISTRIBUTED
    for (int i = 0; i < argc; i++)
        if (strstr(argv[i], "--DFF_GName") != NULL) {
            char *equalPosition = strchr(argv[i], '=');
            groupName = std::string(++equalPosition);
            continue;
        }
    if (DFF_Init(argc, argv) < 0) {
        error("Error while executing: DFF_Init\n");
        return -1;
    }
#endif

    int ncam{7};                // Number of cameras
    int nsqu{1};                // Number of squares (also servers)
    int max_round{-1};          // Number of rounds
    int forcecpu{0};            // Force the execution on the CPU
    std::string data_path;      // Path to the data folder (absolute or with respect to build directory)

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
            if (groupName.compare(loggerName) == 0)
                std::cout
                        << "Usage: mvdet_dist [forcecpu=0/1] [data_path] [num_cameras=7] [num_agg=1] [max_round=-1]\n";
            exit(0);
        } else
            forcecpu = atoi(argv[1]);
    }
    if (argc >= 3)
        data_path = argv[2];
    if (argc >= 4)
        ncam = atoi(argv[3]);
    if (argc >= 5)
        nsqu = atoi(argv[4]);
    if (argc >= 6)
        max_round = (int32_t) atoi(argv[5]);
    if (groupName.compare(loggerName) == 0)
        std::cout << "Inferencing on " << ncam << " cameras." << std::endl;

    torch::DeviceType device_type;
    if (torch::cuda::is_available() && !forcecpu) {
        if (groupName.compare(loggerName) == 0)
            std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        if (groupName.compare(loggerName) == 0)
            std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    torch::cuda::manual_seed_all(42);

    if (groupName.compare(loggerName) == 0)
        std::cout << "Checking data existence..." << std::endl;
    if (!std::filesystem::exists(data_path)) {
        error("Video file cannot be found at path: %s\n", data_path);
        return -1;
    }

    if (groupName.compare(loggerName) == 0)
        std::cout << "Cameras creation..." << std::endl;
    int ncam_x_nsqu = ncam * nsqu;
    Source source(ncam_x_nsqu);
    ControlRoom controlRoom(nsqu);
    ff_a2a a2a;
    std::vector < AggregatorNode * > secondset;
    std::vector < CameraNode * > firstset;
    for (int i = 0; i < nsqu; ++i) {
        std::string id = std::to_string(i + 1);
        secondset.push_back(
                new AggregatorNode("A" + id, data_path + "/map_classifier.pt", data_path + "/coord_map.pt", ncam));
    }
    for (int j = 0; j < nsqu; ++j)
        for (int i = 0; i < ncam; ++i) {
            std::string rank = std::to_string(i + j);
            std::string id = std::to_string(i + j + 1);
            firstset.push_back(new CameraNode("C" + id, data_path + "/Image_subsets_test/C" + id + "/%08d.png",
                                              data_path + "/proj_mat_cam" + rank + ".pt",
                                              data_path + "/base_model.pt",
                                              data_path + "/image_classifier.pt", i, j, max_round));
        }
    source.createGroup("S0");
    for (int i = 0; i < ncam_x_nsqu; i++) {
        std::string id = std::to_string(i + 1);
        a2a.createGroup("W" + id) << firstset[i];
    }
    for (int i = 0; i < nsqu; i++) {
        std::string id = std::to_string(i + 1);
        a2a.createGroup("A" + id) << secondset[i];
    }
    controlRoom.createGroup("S8");
    a2a.add_firstset<CameraNode>(firstset);
    a2a.add_secondset<AggregatorNode>(secondset);


#ifdef DISABLE_FF_DISTRIBUTED
    a2a.wrap_around();
    if (a2a.run_and_wait_end() < 0) {
        error("Error while executing: All-to-All\n");
        return -1;
    }
#else
    ff_pipeline pipe;
    pipe.add_stage(&source);
    pipe.add_stage(&a2a);
    pipe.add_stage(&controlRoom);
    pipe.wrap_around();
    if (pipe.run_and_wait_end() < 0) {
        error("Error while executing: Pipe\n");
        return -1;
    }
#endif

    if (groupName.compare(loggerName) == 0)
        chrono.stop();
    return 0;
}
