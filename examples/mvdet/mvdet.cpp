#include <ff/dff.hpp>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>

#include "utils/net.hpp"
#include "helpers.hpp"

using namespace ff;

struct View {
    View() {}

    explicit View(const uint32_t &s, const uint32_t &f) : id_square{s},
                                                          id_frame{f} {}

    explicit View(View *t) {
        id_square = t->id_square;
        id_frame = t->id_frame;
    }

    torch::Tensor view;
    uint32_t id_square;
    uint32_t id_frame;
};

struct Frame {
    Frame() {}

    explicit Frame(const uint32_t &s, const int32_t &c, const uint32_t &f, const uint32_t &row, const uint32_t &col)
            : id_square{s}, id_camera{c}, id_frame{f}, rows{row}, cols{col} {}

    explicit Frame(const uint32_t &s, const int32_t &c, const uint32_t &f)
            : id_square{s}, id_camera{c}, id_frame{f}, rows{0}, cols{0} {}

    explicit Frame(const uint32_t &s, const uint32_t &f)
            : id_square{s}, id_camera{-1}, id_frame{f}, rows{0}, cols{0} {}

    explicit Frame(Frame *t) {
        id_square = t->id_square;
        id_camera = t->id_camera;
        id_frame = t->id_frame;
        rows = t->rows;
        cols = t->cols;
    }

    //template<class Archive>
    //void serialize(Archive &archive) {
    //    archive(id_square, id_camera, id_frame, matToBytes(frame));
    //} //TODO: serialize frame (già come tensore)

    cv::Mat frame;
    uint32_t id_square;
    int32_t id_camera;
    uint32_t id_frame;
    uint32_t rows;
    uint32_t cols;
};

template<class Archive>
void save(Archive &archive, Frame const &f) {
    std::ostringstream oss;
    uint32_t size = f.frame.total() * f.frame.elemSize();
    uint32_t channels = f.frame.channels();

    oss.write((const char *) &f.id_square, sizeof(f.id_square));
    oss.write((const char *) &f.id_camera, sizeof(f.id_camera));
    oss.write((const char *) &f.id_frame, sizeof(f.id_frame));
    oss.write((const char *) &f.rows, sizeof(f.rows));
    oss.write((const char *) &f.cols, sizeof(f.cols));
    oss.write((const char *) &channels, sizeof(channels));
    oss.write((const char *) &size, sizeof(size));
    oss.write((const char *) f.frame.data, size);
    archive(oss.str());
}

template<class Archive>
void load(Archive &archive, Frame &f) {
    std::string oss;
    archive(oss);
    std::istringstream iss(oss);
    uint32_t size;
    uint32_t channels;

    iss.read((char *) &f.id_square, sizeof(f.id_square));
    iss.read((char *) &f.id_camera, sizeof(f.id_camera));
    iss.read((char *) &f.id_frame, sizeof(f.id_frame));
    iss.read((char *) &f.rows, sizeof(f.rows));
    iss.read((char *) &f.cols, sizeof(f.cols));
    iss.read((char *) &channels, sizeof(channels));
    iss.read((char *) &size, sizeof(size));
    f.frame = cv::Mat(f.rows, f.cols, CV_8UC(channels)).clone();
    iss.read((char *) f.frame.data, size);
}

template<class Archive>
void save(Archive &archive, View const &f) {
    std::ostringstream oss;

    oss.write((const char *) &f.id_square, sizeof(f.id_square));
    oss.write((const char *) &f.id_frame, sizeof(f.id_frame));

    std::ostringstream stream;
    torch::save(f.view, stream);
    std::string buffer = stream.str();
    uint32_t size = buffer.length();
    oss << size << buffer;

    archive(oss.str());
}

template<class Archive>
void load(Archive &archive, View &f) {
    std::string oss;
    archive(oss);
    std::istringstream iss(oss);
    uint32_t size;

    iss.read((char *) &f.id_square, sizeof(f.id_square));
    iss.read((char *) &f.id_frame, sizeof(f.id_frame));
    iss.read((char *) &size, sizeof(size));

    char *buffer = new char[size];
    iss.read(buffer, size);

    std::string string_buffer(buffer, size);
    // Creare uno string buffer è necessario perché char* torch load lo interpreta come filename
    // TODO: fare overloading di torch::load?
    std::istringstream stream(string_buffer);
    torch::load(f.view, stream);

    delete buffer;
}

template<typename T>
void serializefreetask(T *o, Frame *input) {}


struct Source : ff_node_t<int> {
private:
    int round = -1;
    int n_tot_cameras;
public:
    Source() = delete;

    Source(const int n_tot_cameras) : n_tot_cameras{n_tot_cameras} {}

    int *svc(int *i) {
        //sleep(1);
        round++;
        std::cout << "Source starting round " << round << std::endl;
        // Send wakeup to each camera to process next frame
        for (int i = 0; i < n_tot_cameras; i++)
            ff_send_out(new int(round));

        if (i < FF_TAG_MIN) delete i;
        return this->GO_ON;
    }
};

struct CameraNode : ff_monode_t<int, Frame> {
private:
    int out_node, lid;
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
               std::string cm, int lid, int out_node) : camera_id{camera_id}, video_path{vp},
                                                        projection_matrix_path{pm}, base_model_path{bm},
                                                        image_classifier_path{cm},
                                                        out_node{out_node}, lid{lid} {}

    int svc_init() {
        // Load torch  models
        std::cout << "[ Camera " << camera_id << " ] Loading base model (" << base_model_path << ") image classifier ("
                  << image_classifier_path << ") projection matrix (" << projection_matrix_path << std::endl;

        // perspective matrix [3x3]
        torch::jit::script::Module container = torch::jit::load(projection_matrix_path);
        torch::Tensor pm = container.attr("data").toTensor();
        perspective_matrix = tensorToProjectionMat(pm).clone();

        // Model loading
        base_model = new Net<torch::jit::Module>(base_model_path);
        img_classifier = new Net<torch::jit::Module>(image_classifier_path);

        // Opening video
        std::cout << "[ Camera " << camera_id << " ] Starting to process video..." << video_path << std::endl;
        cap = cv::VideoCapture(video_path);

        // Check if video was found
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            return -1;
        }
        return 0;
    }

    Frame *svc(int *i) {
        std::cout << camera_id << " round " << *i << " " << video_path << std::endl;

        // Read next frame
        cap.read(frame);
        if (frame.empty()) {
            std::cout << camera_id << " finished video." << std::endl;
            delete i;
            return this->EOS;
        } else {
            // Process frame using base model
            //show_results(frame, "frame");
            torch::Tensor imgTensor = imgToTensor(frame);
            torch::Tensor img_feature = base_model->forward({imgTensor});

            // Upscaling
            torch::Tensor img_feature_upscaled = torch::nn::functional::interpolate(
                    img_feature,
                    torch::nn::functional::InterpolateFuncOptions()
                            .mode(torch::kBilinear)
                            .size(std::vector<int64_t>({270, 480}))
            );

            // Image classifier
            torch::Tensor img_res = img_classifier->forward({img_feature_upscaled});

            // Create Frame with buffer for output data
            fr = new Frame(out_node, lid, *i, 120, 360);

            // Warp perspective 
            cv::Mat img_feature_mat = tensorToFeat(img_feature, 255);
            //show_results(tensorToFeat(img_feature, 255), "feature_map");
            cv::warpPerspective(img_feature_mat, fr->frame, perspective_matrix, {360, 120});
            //show_results(fr->frame, "warp result");
            //for (int k = 0; k < 10; k++)
            //    printf("%02X", *(fr->frame.data + k));

            // Send it out
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
    std::vector<Frame *> buffer;
    torch::TensorList tensors;
    std::string map_classifier_path;
    std::string coord_mat_path;
    torch::Tensor coord_tensor;
    Net <torch::jit::Module> *map_classifier;
public:
    AggregatorNode() = delete;

    // tensors(nullptr, n_cameras)
    AggregatorNode(std::string id, std::string mm, std::string cm, int n_cameras) : id{id}, n_cameras{n_cameras},
                                                                    buffer(n_cameras, nullptr),
                                                                    map_classifier_path{mm}, coord_mat_path{cm} {}

    int svc_init() {
        map_classifier = new Net<torch::jit::Module>(map_classifier_path);

        torch::jit::script::Module container = torch::jit::load(coord_mat_path);
        coord_tensor = container.attr("data").toTensor();

        return 0;
    }

    Frame *svc(Frame *f) {
        std::cout << id << " recv: square " << f->id_square << " camera " << f->id_camera << " frame " << f->id_frame
                  << std::endl;
        assert(f->id_camera >= 0 && f->id_camera < n_cameras);

        // Save frame in buffer
        buffer[f->id_camera] = f;
        counter++;

        // Buffer full? Process it
        if (counter >= n_cameras) {
            // TODO: Process frame

            torch::Tensor t0 = featToTensor(buffer[0]->frame);
            torch::Tensor t1 = featToTensor(buffer[1]->frame);
            torch::Tensor t2 = featToTensor(buffer[2]->frame);
            torch::Tensor t3 = featToTensor(buffer[3]->frame);
            torch::Tensor t4 = featToTensor(buffer[4]->frame);
            torch::Tensor t5 = featToTensor(buffer[5]->frame);
            torch::Tensor t6 = featToTensor(buffer[6]->frame);

            //std::cout << buffer[0]->frame.size() << endl;
            //for (int k = 0; k < 10; k++)
            //    printf("%02X", *(f->frame.data + k));

            // Convert Map to Tensor and populate List of Tensors
            // expected tensor shape: torch.Size([1, 512, 120, 360])

            // ...

            // Concatenate list of tensors into one big tensor
            // torch::Tensor world_features_cat = torch::cat(torch::TensorList(world_features), 1);

            torch::Tensor world_features_cat = torch::cat({t0, t1, t2, t3, t4, t5, t6, coord_tensor}, 1);

            // Feed tensor into model
            torch::Tensor view = map_classifier->forward(world_features_cat);

            Frame *fr = new Frame(f->id_square, -1, f->id_frame, 120, 360);

            fr->frame = tensorToImg(view, 1);

            // Send out result TODO
            ff_send_out(fr);

            // Free buffer for next round
            counter = 0;
            for (int i = 0; i < n_cameras; i++) {
                delete buffer[i];
                buffer[i] = nullptr;
            }
        }
        return GO_ON;
    }
};

struct ControlRoom : ff_node_t<Frame, int> {
private:
    int counter = 0;
    int n_aggregators;
public:
    ControlRoom() = delete;

    ControlRoom(int n_aggregators) : n_aggregators{n_aggregators} {}

    int *svc(Frame *f) {
        std::cout << "Sink recv: square " << f->id_square << " frame " << f->id_frame << std::endl;
        // show_results(f->frame, "control room result");
        std::cout << f->frame.rows << "x" << f->frame.cols << " channels: " << f->frame.channels() << std::endl;

        delete f;
        counter++;

        // Received all aggragted results? Start new round
        if (counter >= n_aggregators) {
            counter = 0;
            ff_send_out(new int(0));
        }
        return GO_ON;
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
    std::size_t ncam{7};
    std::size_t nsqu{1};
    std::string image_path = "/mnt/shared/gmittone/FastFederatedLearning/mvdet_data/Image_subsets_test";
    std::string data_path = "/mnt/shared/gmittone/FastFederatedLearning/mvdet_data";

    // Parameters parsing
    if (argc >= 2)
        if (strcmp(argv[1], "-h") == 0) {
            if (groupName.compare(sinkName) == 0)
                std::cout << "Usage: mvdet_ff [num_cameras=7] [num_agg=1] [image_path] [data_path]\n";
            exit(0);
        } else
            ncam = atoi(argv[1]);
    if (argc >= 3)
        nsqu = atoi(argv[2]);
    if (argc >= 4)
        image_path = argv[3];
    if (argc >= 5)
        data_path = argv[4];
    if (groupName.compare(sinkName) == 0)
        std::cout << "Inferencing on " << ncam << " cameras." << std::endl;

    std::size_t ncam_x_nsqu{ncam * nsqu};

    // ---- FastFlow components creation -------
    Source source(ncam_x_nsqu);
    ControlRoom controlRoom(nsqu);
    ff_pipeline pipe;
    ff_a2a a2a;

    std::vector < AggregatorNode * > secondset;
    for (int i = 0; i < nsqu; i++) {
        std::string id = std::to_string(i + 1);
        secondset.push_back(new AggregatorNode("A" + id, data_path + "/map_classifier.pt", data_path + "/coord_map.pt", ncam));
    }

    std::vector < CameraNode * > firstset;
    for (int j = 0; j < nsqu; j++)
        for (int i = 0; i < ncam; i++) {
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