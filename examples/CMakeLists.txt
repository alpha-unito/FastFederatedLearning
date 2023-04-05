#     --- Examples's CMake file ---

# Simple single-node master-worker MLP-MNIST example
add_executable(mnist mnist.cpp)
target_link_libraries(mnist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(mnist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(mnist PRIVATE DISABLE_FF_DISTRIBUTED)
configure_file(mnist.json mnist.json COPYONLY)
set_property(TARGET mnist PROPERTY CXX_STANDARD 17)

# Simple single-node peer-to-peer MLP-MNIST example
add_executable(mnist_p2p mnist_p2p.cpp)
target_link_libraries(mnist_p2p "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(mnist_p2p PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(mnist_p2p PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET mnist_p2p PROPERTY CXX_STANDARD 17)

# Distributed master-worker MLP-MNIST example
add_executable(mnist_dist mnist.cpp)
target_link_libraries(mnist_dist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(mnist_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET mnist_dist PROPERTY CXX_STANDARD 17)

# Distributed peer-to-peer MLP-MNIST example
add_executable(mnist_p2p_dist mnist_p2p.cpp)
target_link_libraries(mnist_p2p_dist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(mnist_p2p_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET mnist_p2p_dist PROPERTY CXX_STANDARD 17)
