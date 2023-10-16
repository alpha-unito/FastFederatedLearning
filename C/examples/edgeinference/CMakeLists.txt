#     --- mandown CMake file ---

# Simple shared-memory tree-based inference example
add_executable(edgeinference edgeinference.cpp $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>/C/utils/process-frame.cpp)
target_link_libraries(edgeinference "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
target_include_directories(edgeinference PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(edgeinference PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET edgeinference PROPERTY CXX_STANDARD 17)

# Distributed-memory tree-based inference example
add_executable(edgeinference_dist edgeinference.cpp $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>/C/utils/process-frame.cpp)
target_link_libraries(edgeinference_dist "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
target_include_directories(edgeinference_dist PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
configure_file(edgeinference_2.json edgeinference_2.json COPYONLY)
configure_file(edgeinference_4.json edgeinference_4.json COPYONLY)
configure_file(edgeinference_7.json edgeinference_7.json COPYONLY)
set_property(TARGET edgeinference_dist PROPERTY CXX_STANDARD 17)
if (MPI_FOUND)
    target_link_libraries(edgeinference_dist MPI::MPI_CXX)
    target_include_directories(edgeinference_dist PRIVATE ${MPI_INCLUDE_PATH})
endif ()

