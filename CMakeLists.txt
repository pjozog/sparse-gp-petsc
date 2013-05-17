cmake_minimum_required (VERSION 2.6)

project (sparse-gp-petsc)

#These variables will be inherited in child directories
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set (SRC_DIR "${CMAKE_SOURCE_DIR}/src")

set (CMAKE_C_FLAGS "-Wall -Wundef")
set (CMAKE_CXX_FLAGS "-Wall -Wundef")

# packages
find_package (PETSc REQUIRED)
find_package (MPI REQUIRED)
find_package (GSL REQUIRED)

# c flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PETSC_DEFINITIONS}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")

# interactive options
option (BUILD_SHARED_LIBS "Build shared libraries?" OFF)

set (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
if (NOT IS_DIRECTORY ${LIBRARY_OUTPUT_PATH})
  file (MAKE_DIRECTORY ${LIBRARY_OUTPUT_PATH})
endif ()

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
if (NOT IS_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
  file (MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif ()

set (INCLUDE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/include)
if (NOT IS_DIRECTORY ${INCLUDE_OUTPUT_PATH})
  file (MAKE_DIRECTORY ${INCLUDE_OUTPUT_PATH})
endif ()

set (PKG_CONFIG_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/pkgconfig)
if (NOT IS_DIRECTORY ${PKG_CONFIG_OUTPUT_PATH})
  file (MAKE_DIRECTORY ${PKG_CONFIG_OUTPUT_PATH})
endif ()

# And add the source...
add_subdirectory (src)
