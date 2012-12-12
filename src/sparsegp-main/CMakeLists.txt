add_executable (sparsegp-main main.c)
include_directories (${SRC_DIR} ${PETSC_INCLUDES} ${GSL_INCLUDE_DIR})
target_link_libraries (sparsegp-main ${PETSC_LIBRARIES} ${MPI_C_LIBRARIES} ${GSL_LIBRARIES} sparse-gp)
