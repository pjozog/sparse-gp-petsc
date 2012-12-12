# SOURCE FILES
set (SRC
  sparsegp.h                   sparsegp.c
  io_util.h                    io_util.c
  petsc_util.h                 petsc_util.c
  sparsekernels.h              sparsekernels.c
  gsl_util_matrix.h            gsl_util_matrix.c
  gsl_util_vector.h            gsl_util_vector.c
  gsl_util_blas.h
)

# BUILD LIBRARY
add_library (sparse-gp ${SRC})
include_directories (${PETSC_INCLUDES} ${GSL_INCLUDE_DIR})
target_link_libraries (sparse-gp m ${GSL_LIBRARIES} ${PETSC_LIBRARIES} ${MPI_C_LIBRARIES})
