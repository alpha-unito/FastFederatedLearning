#     --- mandown CMake file ---

# Simple shared-memory tree-based inference example
add_executable(edgeinference edgeinference.cpp process-frame.cpp)
target_compile_definitions(edgeinference PRIVATE DISABLE_FF_DISTRIBUTED)
set_property(TARGET edgeinference PROPERTY CXX_STANDARD 17)
target_link_libraries(edgeinference "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")

# Distributed-memory tree-based inference example
add_executable(edgeinference_dist edgeinference.cpp process-frame.cpp)
target_compile_definitions(edgeinference_dist PRIVATE EXCLUDE_MPI)
target_link_libraries(edgeinference_dist "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}" )
set_property(TARGET edgeinference_dist PROPERTY CXX_STANDARD 17)
