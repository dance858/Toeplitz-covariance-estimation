cmake_minimum_required(VERSION 3.15)
project(NML VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)

# --- Build type default ---
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Platform-specific BLAS/LAPACK ---
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    set(BLAS_LAPACK_LIBS ${ACCELERATE_FRAMEWORK})
    set(HAS_LAPACKE FALSE)
elseif(MSVC)
    find_package(OpenBLAS CONFIG REQUIRED)
    set(BLAS_LAPACK_LIBS OpenBLAS::OpenBLAS)
    set(HAS_LAPACKE FALSE)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_library(LAPACKE_LIB NAMES lapacke)
    if(LAPACKE_LIB)
        set(BLAS_LAPACK_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIB})
        set(HAS_LAPACKE TRUE)
    else()
        set(BLAS_LAPACK_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        set(HAS_LAPACKE FALSE)
    endif()
endif()

# --- FFTW3 ---
if(APPLE)
    execute_process(
        COMMAND brew --prefix fftw
        OUTPUT_VARIABLE FFTW_BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(FFTW_BREW_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH ${FFTW_BREW_PREFIX})
    endif()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
endif()
if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIB NAMES fftw3)
    find_path(FFTW3_INCLUDE fftw3.h)
    if(FFTW3_LIB AND FFTW3_INCLUDE)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE})
    else()
        message(FATAL_ERROR "FFTW3 not found. Install via: brew install fftw (macOS) or apt install libfftw3-dev (Linux)")
    endif()
endif()

# --- NML library ---
set(NML_SOURCES
    c/src/NML_solver.c
    c/src/Newton.c
    c/src/linalg.c
    c/src/levinson_durbin.c
    c/src/diff.c
    c/src/utils.c
)
if(NOT HAS_LAPACKE)
    list(APPEND NML_SOURCES c/src/lapacke_compat.c)
endif()

add_library(nml ${NML_SOURCES})
set_property(TARGET nml PROPERTY POSITION_INDEPENDENT_CODE ON)

if(HAS_LAPACKE)
    target_compile_definitions(nml PUBLIC NML_HAS_LAPACKE)
endif()

target_include_directories(nml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/c>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/c/include
        ${FFTW3_INCLUDE_DIRS}
)

if(FFTW3_LIBRARY_DIRS)
    target_link_directories(nml PUBLIC ${FFTW3_LIBRARY_DIRS})
endif()
target_link_libraries(nml PUBLIC ${BLAS_LAPACK_LIBS} ${FFTW3_LIBRARIES})

if(NOT MSVC)
    target_link_libraries(nml PUBLIC m)
    target_compile_options(nml PRIVATE -Wall -O3)
endif()

# --- Python (scikit-build-core) ---
if(SKBUILD)
    set(BUILD_SHARED_LIBS ON)
    install(TARGETS nml LIBRARY DESTINATION nml)
endif()

# --- Optional: Tests ---
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(c/tests)
endif()

# --- Optional: MATLAB MEX ---
option(BUILD_MATLAB "Build MATLAB MEX interface" OFF)
if(BUILD_MATLAB)
    add_subdirectory(matlab)
endif()
