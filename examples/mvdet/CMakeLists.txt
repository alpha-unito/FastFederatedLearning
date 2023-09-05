#     --- mandown CMake file ---

# Simple shared-memory tree-based inference example
add_executable(mvdet mvdet.cpp helpers.cpp)
target_compile_definitions(mvdet PRIVATE DISABLE_FF_DISTRIBUTED)
target_link_libraries(mvdet "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
target_include_directories(mvdet PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
set_property(TARGET mvdet PROPERTY CXX_STANDARD 17)

# Distributed-memory tree-based inference example
add_executable(mvdet_dist mvdet.cpp helpers.cpp)
target_compile_definitions(mvdet_dist PRIVATE EXCLUDE_MPI)
target_link_libraries(mvdet_dist "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
target_include_directories(mvdet_dist PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
configure_file(mvdet_7_1.json mvdet_7_1.json COPYONLY)
configure_file(mvdet_1_1.json mvdet_1_1.json COPYONLY)
configure_file(mvdet_3_2.json mvdet_3_2.json COPYONLY)
set_property(TARGET mvdet_dist PROPERTY CXX_STANDARD 17)
if (MPI_FOUND)
  target_link_libraries(mvdet_dist MPI::MPI_CXX)
  target_include_directories(mvdet_dist  PRIVATE ${MPI_INCLUDE_PATH})
endif()

