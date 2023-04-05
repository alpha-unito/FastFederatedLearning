#     --- mnist CMake file ---

# Simple shared-memory master-worker MLP-MNIST example
add_executable(mnist mnist.cpp)
target_link_libraries(mnist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(mnist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(mnist PRIVATE DISABLE_FF_DISTRIBUTED)
configure_file(mnist.json mnist.json COPYONLY)
set_property(TARGET mnist PROPERTY CXX_STANDARD 17)

# Distributed master-worker MLP-MNIST example
add_executable(mnist_dist mnist.cpp)
target_link_libraries(mnist_dist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(mnist_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET mnist_dist PROPERTY CXX_STANDARD 17)

