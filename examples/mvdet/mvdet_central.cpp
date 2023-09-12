#include <ff/dff.hpp>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>

#include "utils/net.hpp"
#include "helpers.hpp"
#include "data_stucts.hpp"

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
        for (int i = 0; i < n_tot_cameras; i++)
            ff_send_out(new int(round));

        if (i < FF_TAG_MIN) delete i;
        return this->GO_ON;
    }
};

struct CameraNode : ff_monode_t<int, Frame> {
private:
    int out_node;
    int lid;
    std::string camera_id;
    std::string video_path;
    cv::VideoCapture cap;
public:
    CameraNode() = delete;

    CameraNode(std::string camera_id, std::string vp, std::string pm, std::string bm,
               std::string cm, int lid, int out_node) : camera_id{camera_id}, video_path{vp},
                                                        out_node{out_node}, lid{lid} {}

    int svc_init() {
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

        Frame *fr = new Frame(out_node, lid, *i, 1.0);
        cap.read(fr->frame);

        delete i;

        if (fr->frame.empty()) {
            std::cout << "[ Camera " << camera_id << " ] Finished video." << std::endl;
            delete fr;
            return this->EOS;
        } else {
            // Send it out
            this->ff_send_out_to(fr, out_node);
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
    std::vector<torch::Tensor> buffer;
    torch::TensorList tensors;
    std::string map_classifier_path;
    std::string coord_mat_path;
    std::string projection_matrix_path;
    std::string base_model_path;
    std::string image_classifier_path;
    torch::Tensor coord_tensor;
    Net <torch::jit::Module> *map_classifier;
    std::vector<cv::Mat> perspective_matrices;
    Net <torch::jit::Module> *base_model;
    Net <torch::jit::Module> *img_classifier;
public:
    AggregatorNode() = delete;

    // tensors(nullptr, n_cameras)
    AggregatorNode(std::string id, int n_cameras, std::string mm, std::string cm, std::string pm, std::string bm, std::string ccm) :
                                                                                    id{id}, n_cameras{n_cameras},
                                                                                    buffer(n_cameras+1),
                                                                                    perspective_matrices(n_cameras),
                                                                                    map_classifier_path{mm},
                                                                                    coord_mat_path{cm},
                                                                                    projection_matrix_path{pm},
                                                                                    base_model_path{bm},
                                                                                    image_classifier_path{ccm} {}

    int svc_init() {
        // perspective matrices for each camera [3x3]
        for(int i = 0; i < n_cameras; i++) {
            int size_s = std::snprintf(nullptr, 0, projection_matrix_path.c_str(), i) + 1; // Extra space for '\0'
            if(size_s <= 0) {
                throw std::runtime_error("Error during formatting.");
            }
            auto size = static_cast<size_t>(size_s);
            std::unique_ptr<char[]> buf( new char[ size ] );
            std::snprintf(buf.get(), size, projection_matrix_path.c_str(), i);
            std::string path(buf.get(), buf.get() + size - 1);
            
            std::cout << "[ Aggregator " << id << " ] Loading projection matrix: " << path << std::endl;
            torch::jit::script::Module container = torch::jit::load(path);
            torch::Tensor pm = container.attr("data").toTensor();
            perspective_matrices[i] = tensorToProjectionMat(pm).clone();
        }

        // Model loading
        std::cout << "[ Aggregator " << id << " ] Loading base model: " << base_model_path << std::endl;
        base_model = new Net<torch::jit::Module>(base_model_path);
        // std::cout << "[ Aggregator " << id << " ] Loading image classifier : " << image_classifier_path << std::endl;
        // img_classifier = new Net<torch::jit::Module>(image_classifier_path);
        std::cout << "[ Aggregator " << id << " ] Loading map classifier: " << map_classifier_path << std::endl;
        map_classifier = new Net<torch::jit::Module>(map_classifier_path);

        // Last buffer entry contains fixed coordinate matrix
        std::cout << "[ Aggregator " << id << " ] Loading coord matrix: " << coord_mat_path << std::endl;
        torch::jit::script::Module container = torch::jit::load(coord_mat_path);
        buffer[n_cameras] = container.attr("data").toTensor();

        return 0;
    }

    Frame *svc(Frame *f) {
        std::cout << "[ Aggregator " << id << " ] Received square " << f->id_square << " camera " << f->id_camera << " frame " << f->id_frame
                  << std::endl;
        assert(f->id_camera >= 0 && f->id_camera < n_cameras);

        // Process frame using base model
        // show_results(f->frame, "frame");
        torch::Tensor imgTensor = imgToTensor(f->frame);
        // show_results(imgTensor, "tensor");
        imgTensor[0][0] = imgTensor[0][0].sub(0.485).div(0.229);
        imgTensor[0][1] = imgTensor[0][1].sub(0.456).div(0.224);
        imgTensor[0][2] = imgTensor[0][2].sub(0.406).div(0.225);
        // show_results(imgTensor, "normalised tensor");
        
        torch::Tensor img_feature = base_model->forward({imgTensor});
        float img_feature_max = img_feature.max().item().toFloat();

        // Warp perspective and save into buffer[f->id_camera]
        cv::Mat img_feature_mat = tensorToImg(img_feature);
        cv::Mat img_feature_mat_warped;
        cv::warpPerspective(img_feature_mat, img_feature_mat_warped, perspective_matrices[f->id_camera], {360, 120});

        buffer[f->id_camera] = featToTensor(img_feature_mat_warped, img_feature_max);

        // Increase camera counter
        counter++;

        // Buffer full? Process it
        if (counter >= n_cameras) {
            //std::cout << buffer[0]->frame.size() << endl;
            //for (int k = 0; k < 10; k++)
            //    printf("%02X", *(f->frame.data + k));

            // Convert Map to Tensor and populate List of Tensors
            // expected tensor shape: torch.Size([1, 512, 120, 360])

            // ...

            // Concatenate list of tensors into one big tensor
            // torch::Tensor world_features_cat = torch::cat(torch::TensorList(world_features), 1);

            // torch::Tensor world_features_cat = torch::cat({t0, t1, t2, t3, t4, t5, t6, coord_tensor}, 1);
            torch::Tensor world_features_cat = torch::cat(buffer, 1);

            // Feed tensor into model
            torch::Tensor view = map_classifier->forward(world_features_cat);
            float view_max = view.max().item().toFloat();

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

            Frame *fr = new Frame(f->id_square, -1, f->id_frame);

            fr->frame = tensorToImg(view);
            //show_results(fr->frame, "aggregator result");

            // Send out result TODO
            ff_send_out(fr);

            // Free buffer for next round
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
    std::vector<std::chrono::steady_clock::time_point*> starts;
    std::vector<std::chrono::steady_clock::time_point> ends;
public:
    ControlRoom() = delete;

    ControlRoom(int n_aggregators) : n_aggregators{n_aggregators}, starts(n_aggregators, nullptr), ends(n_aggregators) {}

    int *svc(Frame *f) {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        if(starts[f->id_square] == nullptr) {
            starts[f->id_square] = new std::chrono::steady_clock::time_point(now);
            std::cout << "[ Control Room ] Received first view " << f->id_frame << " from square " << f->id_square << std::endl;
        } else {
            double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - ends[f->id_square]).count();
            std::cout << "[ Control Room ] Received new view " << f->id_frame << " from square " << f->id_square << " (processing time " << elapsed_ms/1000.0 << " s)" << std::endl;
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
        if(tot_counter >= 2) {
            std::chrono::steady_clock::time_point start;
            std::chrono::steady_clock::time_point end;

            for(int i; i < n_aggregators; i++) {
                if(starts[i] != nullptr) {
                    start = *(starts[i]);
                    end = ends[i];
                    break; //TODO better way
                }
            }


            for(int i; i < n_aggregators; i++) {
                if(starts[i] != nullptr && *(starts[i]) < start) start = *(starts[i]);
                if(ends[i] > end) end = ends[i];
            }

            double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            uint64_t processed_views = tot_counter - 1;
            
            std::cout << "[ Control Room ] Processed " << processed_views << " views in " << elapsed_ms/1000.0 << " s ( " << elapsed_ms / 1000.0 /  processed_views << " s/view)" << std::endl;
        }
    }
};

int main(int argc, char *argv[]) {
    std::string groupName = "S0";
    std::string sinkName = "S0";

#ifndef DISABLE_FF_DISTRIBUTED
    // distributed RTS init ------
    for (int i = 0; i < argc; i++)
        if (strstr(argv[i], "--DFF_GName") != NULL) {
            char *equalPosition = strchr(argv[i], '=');
            groupName = std::string(++equalPosition);
            continue;
        }
    if (DFF_Init(argc, argv) < 0) {
        error("DFF_Init\n");
        return -1;
    }
#endif
    // then this could be std::vector
    uint32_t ncam{7};
    uint32_t nsqu{1};
    std::string image_path = "/mnt/shared/gmittone/FastFederatedLearning/mvdet_data/Image_subsets_sequencial";
    std::string data_path = "/mnt/shared/gmittone/FastFederatedLearning/mvdet_data";

    // Parameters parsing
    if (argc >= 2)
        if (strcmp(argv[1], "-h") == 0) {
            if (groupName.compare(sinkName) == 0)
                std::cout << "Usage: mvdet_ff [num_cameras=7] [num_agg=1] [image_path] [data_path]\n";
            exit(0);
        } else
            ncam = (uint32_t) atoi(argv[1]);
    if (argc >= 3)
        nsqu = (uint32_t) atoi(argv[2]);
    if (argc >= 4)
        image_path = argv[3];
    if (argc >= 5)
        data_path = argv[4];
    if (groupName.compare(sinkName) == 0)
        std::cout << "Inferencing on " << ncam << " cameras." << std::endl;

    torch::cuda::manual_seed_all(42);

    std::size_t ncam_x_nsqu{ncam * nsqu};

    // ---- FastFlow components creation -------
    Source source(ncam_x_nsqu);
    ControlRoom controlRoom(nsqu);
    ff_pipeline pipe;
    ff_a2a a2a;

    std::vector < AggregatorNode * > secondset;
    for (uint32_t i = 0; i < nsqu; i++) {
        std::string id = std::to_string(i + 1);
        secondset.push_back(
                new AggregatorNode("A" + id, ncam, data_path + "/map_classifier.pt", data_path + "/coord_map.pt", 
                                              data_path + "/proj_mat_cam%d.pt",
                                              data_path + "/base_model.pt",
                                              data_path + "/image_classifier.pt"));
    }

    std::vector < CameraNode * > firstset;
    for (uint32_t j = 0; j < nsqu; j++)
        for (uint32_t i = 0; i < ncam; i++) {
            std::string rank = std::to_string(i + j);
            std::string id = std::to_string(i + j + 1);
            firstset.push_back(new CameraNode("C" + id, image_path + "/C" + id + "/%08d.png",
                                              data_path + "/proj_mat_cam" + rank + ".pt",
                                              data_path + "/base_model.pt",
                                              data_path + "/image_classifier.pt", i, j));
        }

    // ---- FastFlow graph -------
    a2a.add_firstset<CameraNode>(firstset);
    a2a.add_secondset<AggregatorNode>(secondset);
    pipe.add_stage(&source);
    pipe.add_stage(&a2a);
    pipe.add_stage(&controlRoom);
    pipe.wrap_around();

    // --- distributed groups ----
    source.createGroup("S0");
    for (int i = 0; i < ncam_x_nsqu; i++) {
        std::string id = std::to_string(i + 1);
        a2a.createGroup("C" + id) << firstset[i];
    }
    for (int i = 0; i < nsqu; i++) {
        std::string id = std::to_string(i + 1);
        a2a.createGroup("A" + id) << secondset[i];
    }
    controlRoom.createGroup("S8");

    if (pipe.run_and_wait_end() < 0) {
        error("running the main pipe\n");
        return -1;
    }
    return 0;
}
