cmake_minimum_required(VERSION 3.12)
project(NML_source)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-comment -Wno-implicit-function-declaration -O3")


find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Check if CBLAS header is found
include(CheckIncludeFile)
check_include_file(cblas.h HAS_CBLAS_H)
if(NOT HAS_CBLAS_H)
    message(FATAL_ERROR "CBLAS not found. Please install CBLAS. ")
endif()

# Check if LAPACKE is found
find_library(LAPACKE_LIB NAMES lapacke)
if(NOT LAPACKE_LIB)
  message(FATAL_ERROR "LAPACKE not found. Please install LAPACKE.")
endif()

add_library(${PROJECT_NAME}
  src/diff.c
  src/levinson_durbin.c
  src/linalg.c
  src/Newton.c
  src/NML_solve.c
  src/utils.c
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  )

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${BLAS_LIBRARIES}
  fftw3
  ${LAPACK_LIBRARIES}
)

if(BUILD_MATLAB_COMMANDS)
    find_package(Matlab)
    matlab_add_mex(NAME NML SRC NML_mex.c LINK_TO ${PROJECT_NAME})
    message(STATUS "MATLAB-related commands will be built.")
else()
    message(STATUS "MATLAB-related commands will not be built.")
endif()
