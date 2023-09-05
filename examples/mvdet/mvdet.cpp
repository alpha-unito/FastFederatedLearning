#include <ff/dff.hpp>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <filesystem>

#include "utils/net.hpp"

#include "helpers.hpp"

using namespace ff;

struct Frame {
    Frame() {}

    explicit Frame(const int &s, const int &c, const int &f) : id_square{s}, id_camera{c}, id_frame{f} {}

    explicit Frame(const int &s, const int &f) : id_square{s}, id_camera{-1}, id_frame{f} {}

    explicit Frame(Frame *t) {
        id_square = t->id_square;
        id_camera = t->id_camera;
        id_frame = t->id_frame;
    }

    template<class Archive>
    void serialize(Archive &archive) {
        archive(id_square, id_camera, id_frame);
    }

    cv::Mat frame;
    int id_square;
    int id_camera;
    int id_frame;
};

struct Source : ff_node_t<int> {
    Source(int n_tot_cameras) : n_tot_cameras{n_tot_cameras} {}

    int *svc(int *i) {
        sleep(1); // TODO: remove sleep
        round++;
        std::cout << "Source starting round " << round << std::endl;
        // Send wakeup to each camera to process next frame
        for (int i = 0; i < n_tot_cameras; i++)
            ff_send_out(new int(round));

        if (i < FF_TAG_MIN) delete i;
        return GO_ON;
    }

    int round = 0;
    int n_tot_cameras;
};

struct CameraNode : ff_monode_t<int, Frame> {
    CameraNode(std::string camera_id, std::string vp, std::string pm, std::string bm,
            std::string cm, int lid, int out_node) : camera_id{camera_id},
                                                                                         video_path{vp},
                    projection_matrix_path{pm}, base_model_path{bm}, image_classifier_path{cm},
            out_node{out_node}, lid{lid} {}

    int svc_init() {

        // Load torch  models
        std::cout << "[ Camera " << camera_id << " ] Loading base model (" << base_model_path << ") image classifier (" << image_classifier_path
                << ") projection matrix (" << projection_matrix_path <<std::endl;
        
        // perspective matrix [3x3]
        torch::jit::script::Module container = torch::jit::load(projection_matrix_path);
        
        torch::Tensor pm = container.attr("data").toTensor();
        perspective_matrix = tensorToProjectionMat(pm);

        // base model
        base_model = new Net<torch::jit::Module>(base_model_path);

        // image classifier
        img_classifier = new Net<torch::jit::Module>(image_classifier_path);

        std::cout << "[ Camera " << camera_id << " ] Starting to process video..." << video_path << std::endl;

        // Opening video
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
            std::cout << "Finish Video\n";
            return EOS;
        } else {

            // Process frame using base model
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
            Frame * fr = new Frame(out_node, lid, *i);

            // Warp perspective 
            cv::Mat img_feature_mat =  tensorToImg(img_feature);
            cv::warpPerspective(img_feature_mat, fr->frame, perspective_matrix, {120, 360});
    
            // Send it out
            ff_send_out_to(fr, out_node);
            delete i;
            return GO_ON;
        }
    }

    void svc_end() {
        cap.release();
        std::cout << "[ Camera " << camera_id << " ] Video closed, process finished. " << std::endl;
    }

    //----------------------------------------------------------------------------------------------------------
    int out_node, lid;
    std::string camera_id;
    std::string video_path;
    std::string projection_matrix_path;
    std::string base_model_path;
    std::string image_classifier_path;
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Mat perspective_matrix;
    Net <torch::jit::Module> * base_model;
    Net <torch::jit::Module> * img_classifier;
};

struct AggregatorNode : ff_minode_t<Frame> {
    AggregatorNode(std::string id, std::string mm, int n_cameras) : id{id},  n_cameras{n_cameras},  buffer(n_cameras, nullptr),
        map_classifier_path{mm} {}

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

            /*
			    at::tensor world_features_cat = torch::cat(torch::TensorList(world_features), 1);
			    at::tensor map_result = map_classifier.forward(world_features_cat);
            
            */
            // Send out result
            ff_send_out(new Frame(f->id_square, f->id_frame));

            // Free buffer for next round
            counter = 0;
            for (int i = 0; i < n_cameras; i++) {
                delete buffer[i];
                buffer[i] = nullptr;
            }
        }

        return GO_ON;
    }

    int counter = 0;
    int n_cameras;
    std::string id;
    std::vector<Frame *> buffer;
    
    std::string map_classifier_path;
};

struct ControlRoom : ff_node_t<Frame, int> {
    ControlRoom(int n_aggregators) : n_aggregators{n_aggregators} {}

    int *svc(Frame *f) {
        std::cout << "Sink recv: square " << f->id_square << " camera " << f->id_camera << " frame " << f->id_frame
                  << std::endl;
        delete f;

        counter++;

        // Received all aggragted results? Start new round
        if (counter >= n_aggregators) {
            counter = 0;
            ff_send_out(new int(0));
        }

        return GO_ON;
    }

    int counter = 0;
    int n_aggregators;
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

    std::string image_path = "/home/gmalenza/Workspace/Phd/mytest2/Image_subsets_test";

    // then this could be std::vector
    std::size_t ncam{7};
    std::size_t nsqu{1};
    std::string image_path = "/mnt/shared/gmittone/23_dml_icc/Image_subsets_test";
    std::string base_models_path = "/mnt/home/birke/Data/Wildtrack_Trained_MVDet";

    // Parameters parsing
    if (argc >= 2)
        if (strcmp(argv[1], "-h") == 0) {
            if (groupName.compare(sinkName) == 0)
                std::cout << "Usage: mvdet_ff [num_cameras=7] [num_agg=1] [data_path]\n";
            exit(0);
        } else
            ncam = atoi(argv[1]);
    if (argc >= 3)
        nsqu = atoi(argv[2]);
    if (argc >= 4)
        image_path = argv[3];
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
        std::string rank = std::to_string(i + 1);
        secondset.push_back(new AggregatorNode("A" + rank, base_models_path + "map_classifier.pt", ncam));
        AggregatorNode A1("A1", , ncam_x_nsqu);
    }

    std::vector < CameraNode * > firstset;
    for (int j = 0; j < nsqu; j++)
        for (int i = 0; i < ncam; i++) {
            std::string rank = std::to_string(i + j + 1);
            firstset.push_back(new CameraNode("C" + rank, image_path + "/C" + rank+"/%08d.png",
                base_models_path + "proj_mat_cam" + rank + ".pt", base_models_path + "/base_model.pt",
                base_models_path + "/image_classifier.pt",  i, j));
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
        std::string rank = std::to_string(i + 1);
        a2a.createGroup("C" + rank) << firstset[i];
    }
    for (int i = 0; i < nsqu; i++) {
        std::string rank = std::to_string(i + 1);
        a2a.createGroup("A" + rank) << secondset[i];
    }
    controlRoom.createGroup("S8");

    if (pipe.run_and_wait_end() < 0) {
        error("running the main pipe\n");
        return -1;
    }
    return 0;
}