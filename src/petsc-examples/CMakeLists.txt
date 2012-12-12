add_executable (hello-world hello-world.c)

include_directories (${SRC_DIR} ${PETSC_INCLUDES})

# serial example
add_executable (ksp-ex1 ksp-ex1.c)
target_link_libraries (ksp-ex1 ${PETSC_LIBRARIES} ${MPI_C_LIBRARIES})

# parallel example
add_executable (ksp-ex23 ksp-ex23.c)
target_link_libraries (ksp-ex23 ${PETSC_LIBRARIES} ${MPI_C_LIBRARIES})
