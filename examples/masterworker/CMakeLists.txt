#     --- masterworker CMake file ---

# Simple shared-memory master-worker MLP-masterworker example
add_executable(masterworker masterworker.cpp)
target_link_libraries(masterworker "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT}) 
target_include_directories(masterworker PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_compile_definitions(masterworker PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET masterworker PROPERTY CXX_STANDARD 17)

# Distributed master-worker MLP-masterworker example
add_executable(masterworker_dist masterworker.cpp)
target_link_libraries(masterworker_dist "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(masterworker_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
configure_file(masterworker_2.json masterworker_2.json COPYONLY)
configure_file(masterworker_4.json masterworker_4.json COPYONLY)
configure_file(masterworker_8.json masterworker_8.json COPYONLY)
set_property(TARGET masterworker_dist PROPERTY CXX_STANDARD 17)

