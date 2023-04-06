#     --- mandown CMake file ---

# Simple shared-memory tree-based inference example
add_executable(edgeinference edgeinference.cpp process-frame.cpp)
target_compile_definitions(edgeinference PRIVATE DISABLE_FF_DISTRIBUTED)
target_link_libraries(edgeinference "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
set_property(TARGET edgeinference PROPERTY CXX_STANDARD 17)

# Distributed-memory tree-based inference example
add_executable(edgeinference_dist edgeinference.cpp process-frame.cpp)
target_compile_definitions(edgeinference_dist PRIVATE EXCLUDE_MPI)
target_link_libraries(edgeinference_dist "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
configure_file(edgeinference_2.json edgeinference_2.json COPYONLY)
configure_file(edgeinference_4.json edgeinference_4.json COPYONLY)
configure_file(edgeinference_7.json edgeinference_7.json COPYONLY)
set_property(TARGET edgeinference_dist PROPERTY CXX_STANDARD 17)
