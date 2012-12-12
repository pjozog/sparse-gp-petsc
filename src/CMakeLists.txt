# set (EIGEN3_INCLUDE_DIR "${SRC_DIR}/eigen")
add_definitions (-std=gnu99)

add_subdirectory (sparsegp)
add_subdirectory (petsc-examples)
add_subdirectory (sparsegp-main)
