#     --- Examples's CMake file ---

# Simple shared-memory master-worker MLP-MNIST example
add_executable(mnist mnist/mnist.cpp)
target_link_libraries(mnist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(mnist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(mnist PRIVATE DISABLE_FF_DISTRIBUTED)
configure_file(mnist/mnist.json mnist/mnist.json COPYONLY)
set_property(TARGET mnist PROPERTY CXX_STANDARD 17)

# Simple shared-memory peer-to-peer MLP-MNIST example
add_executable(mnist_p2p mnist_p2p/mnist_p2p.cpp)
target_link_libraries(mnist_p2p "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(mnist_p2p PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(mnist_p2p PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET mnist_p2p PROPERTY CXX_STANDARD 17)

# Simple shared-memory tree-based inference example
add_executable(mandown mandown/mandown.cpp mandown/process-frame.cpp)
target_compile_definitions(mandown PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET mandown PROPERTY CXX_STANDARD 17)
target_link_libraries(mandown "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")

# Distributed master-worker MLP-MNIST example
add_executable(mnist_dist mnist/mnist.cpp)
target_link_libraries(mnist_dist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(mnist_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET mnist_dist PROPERTY CXX_STANDARD 17)

# Distributed peer-to-peer MLP-MNIST example
add_executable(mnist_p2p_dist mnist_p2p/mnist_p2p.cpp)
target_link_libraries(mnist_p2p_dist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(mnist_p2p_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET mnist_p2p_dist PROPERTY CXX_STANDARD 17)

# Distributed-memory tree-based inference example
add_executable(mandown_dist mandown/mandown.cpp mandown/process-frame.cpp)
target_compile_definitions(mandown_dist PRIVATE EXCLUDE_MPI)
target_link_libraries(mandown_dist "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}" )
set_property(TARGET mandown_dist PROPERTY CXX_STANDARD 17)

