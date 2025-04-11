cmake_minimum_required(VERSION 3.22)
project(CoverageSetGeneration LANGUAGES CXX CUDA)

if (UNIX)
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

# for color output only
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

SET(CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/build")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# set the g++ compiler flags
set(CMAKE_CXX_FLAGS "-O3 -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_LOADED)
    message(STATUS "Compiler Path: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# include headers
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# find headers
file(GLOB HEADER_FILES_HPP ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.h)
message(STATUS "Header Files: ${HEADER_FILES_HPP}")

set(RRTZ_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/obs.cpp
    ${CMAKE_SOURCE_DIR}/src/rrtz.cpp
    ${CMAKE_SOURCE_DIR}/src/station.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

set(COV_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/obs.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/cost_matrix.cpp
    ${CMAKE_SOURCE_DIR}/src/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/obs.cpp
    ${CMAKE_SOURCE_DIR}/src/rrtz.cpp
    ${CMAKE_SOURCE_DIR}/src/station.cpp
    ${CMAKE_SOURCE_DIR}/src/tsp.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/viewpoint_generator.cpp
)

# add_executable(station_example ${RRTZ_SOURCES} ${CMAKE_SOURCE_DIR}/examples/station_example.cpp)
# add_executable(cube_example ${RRTZ_SOURCES} ${CMAKE_SOURCE_DIR}/examples/cube_example.cpp)
# add_executable(test_raytracing ${SOURCES} ${CMAKE_SOURCE_DIR}/examples/test_raytracing.cpp ${HEADER_FILES_HPP})

# save models as obj files
add_executable(obj_converter ${SOURCES} ${CMAKE_SOURCE_DIR}/examples/obj_converter.cpp ${HEADER_FILES_HPP})
add_executable(run_ccpp ${SOURCES} ${CMAKE_SOURCE_DIR}/src/run_ccpp.cpp ${HEADER_FILES_HPP})
add_executable(get_coverage ${SOURCES} ${CMAKE_SOURCE_DIR}/examples/evaluate_path_coverage.cpp ${HEADER_FILES_HPP})
add_executable(cw_coverage ${SOURCES} ${CMAKE_SOURCE_DIR}/examples/evaluate_cw_coverage.cpp ${HEADER_FILES_HPP})
add_executable(saturation ${SOURCES} ${CMAKE_SOURCE_DIR}/examples/evaluate_path_saturation.cpp ${HEADER_FILES_HPP})
add_executable(ordered_vp_cost_eval ${SOURCES} ${CMAKE_SOURCE_DIR}/examples/evaluate_ordered_vp_cost.cpp ${HEADER_FILES_HPP})

# add_executable(test_path_collisions ${SOURCES} ${CMAKE_SOURCE_DIR}/src/test_path_collisions.cpp ${HEADER_FILES_HPP})

#link cuda
target_link_libraries(obj_converter PRIVATE ${CUDA_CUDA_LIBRARY})
target_link_libraries(run_ccpp PRIVATE ${CUDA_CUDA_LIBRARY})
target_link_libraries(get_coverage PRIVATE ${CUDA_CUDA_LIBRARY})
target_link_libraries(cw_coverage PRIVATE ${CUDA_CUDA_LIBRARY})
target_link_libraries(saturation PRIVATE ${CUDA_CUDA_LIBRARY})
target_link_libraries(ordered_vp_cost_eval PRIVATE ${CUDA_CUDA_LIBRARY})

# target_link_libraries(test_raytracing PRIVATE ${CUDA_CUDA_LIBRARY})