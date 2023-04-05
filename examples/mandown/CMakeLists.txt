#     --- mandown CMake file ---

# Simple shared-memory tree-based inference example
add_executable(mandown mandown.cpp process-frame.cpp)
target_compile_definitions(mandown PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET mandown PROPERTY CXX_STANDARD 17)
target_link_libraries(mandown "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")

# Distributed-memory tree-based inference example
add_executable(mandown_dist mandown.cpp process-frame.cpp)
target_compile_definitions(mandown_dist PRIVATE EXCLUDE_MPI)
target_link_libraries(mandown_dist "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}" )
set_property(TARGET mandown_dist PROPERTY CXX_STANDARD 17)
