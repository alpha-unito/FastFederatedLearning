### 	FFL - FAST FEDERATED LEARNING 	### 
# 			--- Main CMake file ---
message("###########################################")
message("### 	FFL - FAST FEDERATED LEARNING 	###")
message("###########################################")
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(FFL)

enable_language(Fortran)

# Retrieving libraries paths
list(APPEND CMAKE_PREFIX_PATH $ENV{CEREAL_HOME} $ENV{FF_HOME} $ENV{TORCH_HOME})
message("CMAKE found the following libraries paths:")
foreach(path ${CMAKE_PREFIX_PATH})
  message("- Path: " ${path})
endforeach(path)

# Searching for the FastFlow library
find_path(FASTFLOW_DIR NAMES "ff/ff.hpp")
if(FASTFLOW_DIR STREQUAL "FASTFLOW_DIR-NOTFOUND")
  message(FATAL_ERROR "Could not find FastFlow library.")  
endif()
include_directories(${FASTFLOW_DIR})

# Searching for the Cereal library
find_path(CEREAL_DIR NAMES "cereal/cereal.hpp")
if(CEREAL_DIR STREQUAL "CEREAL_DIR-NOTFOUND")
  message(FATAL_ERROR "Could not find Cereal library.")
endif()
include_directories(${CEREAL_DIR})

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Adding compiler definitions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
add_compile_definitions(DFF_EXCLUDE_MPI)
add_compile_definitions(NO_DEFAULT_MAPPING)

# The example folder contains the actual examples' code
add_subdirectory( examples )
