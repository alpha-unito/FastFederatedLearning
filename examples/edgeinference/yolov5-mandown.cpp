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
#include "process-frame.hpp"

//using json = nlohmann::json;
using namespace std;

int main(int argc, const char *argv[])
{
    

    if (argc < 4) {
        std::cerr << "usage: mandown <path to torchscript file> <path to video> <yes/no>" << std::endl;
        std::cerr << "  if last parameter is yes results will be displayed graphically" << std::endl;
        return -1; 
    }

    if( std::filesystem::exists(argv[1]) == false ) {
        std::cerr << "Torchscript file cannot be found at path: " << argv[1] << std::endl;
        return -1;
    }

    if( std::filesystem::exists(argv[2]) == false ) {
        std::cerr << "Video file cannot be found at path: " << argv[2] << std::endl;
        return -1;
    }

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[1]);
    } catch (const c10::Error &e){
        std::cerr   << "Error loading the model\n" << std::endl
                    << "Details:" << std::endl
                    << e.what() << std::endl;
        return -1;
    }

    process_video(model, argv[2], std::string(argv[3]) == "yes");

    return 0;   
}
