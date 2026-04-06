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
elseif(MSVC)
    find_package(OpenBLAS CONFIG REQUIRED)
    find_package(clapack CONFIG REQUIRED)
    set(BLAS_LAPACK_LIBS OpenBLAS::OpenBLAS lapack)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(BLAS_LAPACK_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
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
    C/src/NML_solver.c
    C/src/Newton.c
    C/src/linalg.c
    C/src/levinson_durbin.c
    C/src/diff.c
    C/src/utils.c
    C/src/lapacke_compat.c
)

add_library(nml ${NML_SOURCES})
set_property(TARGET nml PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(nml PRIVATE NML_VERSION="${PROJECT_VERSION}")

target_include_directories(nml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/C>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/C/include
        ${FFTW3_INCLUDE_DIRS}
)

if(FFTW3_LIBRARY_DIRS)
    target_link_directories(nml PUBLIC ${FFTW3_LIBRARY_DIRS})
endif()
target_link_libraries(nml PUBLIC ${BLAS_LAPACK_LIBS} ${FFTW3_LIBRARIES})

if(MSVC)
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        # clang-cl needs the Clang runtime for complex arithmetic (__divdc3, __muldc3)
        execute_process(
            COMMAND "${CMAKE_C_COMPILER}" --print-runtime-dir
            OUTPUT_VARIABLE CLANG_RT_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE CLANG_RT_RESULT
        )
        if(CLANG_RT_RESULT EQUAL 0 AND EXISTS "${CLANG_RT_DIR}")
            message(STATUS "Clang runtime dir: ${CLANG_RT_DIR}")
        else()
            # Fallback: find it relative to the compiler (VS-bundled Clang)
            get_filename_component(CLANG_BIN_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
            file(GLOB CLANG_RT_DIR "${CLANG_BIN_DIR}/../lib/clang/*/lib/windows")
            if(CLANG_RT_DIR)
                list(GET CLANG_RT_DIR 0 CLANG_RT_DIR)
                message(STATUS "Clang runtime dir (fallback): ${CLANG_RT_DIR}")
            endif()
        endif()
        if(CLANG_RT_DIR AND EXISTS "${CLANG_RT_DIR}")
            target_link_directories(nml PUBLIC "${CLANG_RT_DIR}")
            target_link_libraries(nml PUBLIC clang_rt.builtins-x86_64)
        else()
            message(WARNING "Could not find Clang runtime builtins library")
        endif()
    endif()
else()
    target_link_libraries(nml PUBLIC m)
    target_compile_options(nml PRIVATE -Wall -O3)
endif()

# --- Python (scikit-build-core) ---
if(SKBUILD)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
    python3_add_library(_nml MODULE python/nml/_bindings.c)
    target_include_directories(_nml PRIVATE
        ${CMAKE_SOURCE_DIR}/C
        ${CMAKE_SOURCE_DIR}/C/include
        ${FFTW3_INCLUDE_DIRS}
    )
    target_link_libraries(_nml PRIVATE nml Python3::NumPy)
    install(TARGETS _nml DESTINATION nml)
endif()

# --- Optional: Tests ---
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(C/tests)
endif()

# --- Optional: MATLAB MEX ---
option(BUILD_MATLAB "Build MATLAB MEX interface" OFF)
if(BUILD_MATLAB)
    add_subdirectory(matlab)
endif()
