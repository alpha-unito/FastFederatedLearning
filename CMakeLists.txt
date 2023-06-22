### 	FFL - FAST FEDERATED LEARNING 	### 
# 			--- Main CMake file ---

message("###########################################")
message("### 	FFL - FAST FEDERATED LEARNING 	###")
message("###########################################")
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(FFL-FastFederatedLearning)

# Retrieving libraries paths
list(APPEND CMAKE_PREFIX_PATH $ENV{CEREAL_HOME} $ENV{FF_HOME} $ENV{TORCH_HOME})

# Searching for the FastFlow library
find_path(FASTFLOW_DIR NAMES "ff/ff.hpp")
if (FASTFLOW_DIR STREQUAL "FASTFLOW_DIR-NOTFOUND")
    message(FATAL_ERROR "Could not find FastFlow library.")
endif ()
include_directories(${FASTFLOW_DIR})

# Searching for the Cereal library
find_path(CEREAL_DIR NAMES "cereal/cereal.hpp")
if (CEREAL_DIR STREQUAL "CEREAL_DIR-NOTFOUND")
    message(FATAL_ERROR "Could not find Cereal library.")
endif ()
include_directories(${CEREAL_DIR})

# Searching for Torch and OpenCV
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED core highgui videoio) # TODO if not found OpenCV then exclude edgeinference


find_package(MPI)

if (NOT MPI_FOUND)
    message("DFF excluding MPI")
    add_compile_definitions(DFF_EXCLUDE_MPI)
endif ()

# Adding compiler definitions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
add_compile_definitions(NO_DEFAULT_MAPPING)

# The example folder contains the actual examples' code
add_subdirectory(examples)
