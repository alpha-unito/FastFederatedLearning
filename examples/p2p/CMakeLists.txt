#     --- p2p CMake file ---

# Simple shared-memory peer-to-peer MLP-MNIST example
add_executable(p2p p2p.cpp)
target_link_libraries(p2p "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(p2p PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(p2p PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET p2p PROPERTY CXX_STANDARD 17)

# Distributed peer-to-peer MLP-MNIST example
add_executable(p2p_dist p2p.cpp)
target_link_libraries(p2p_dist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(p2p_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
configure_file(p2p_2.json p2p_2.json COPYONLY)
configure_file(p2p_4.json p2p_4.json COPYONLY)
configure_file(p2p_8.json p2p_8.json COPYONLY)
set_property(TARGET p2p_dist PROPERTY CXX_STANDARD 17)