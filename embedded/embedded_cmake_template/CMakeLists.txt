cmake_minimum_required(VERSION 3.11)
project(main)

include(CMake/ConfigureCrossCompile.cmake)
include(CMake/mlpack.cmake)

option(USE_OPENMP "If available, use OpenMP for parallelization." ON)

fetch_mlpack(ON)

add_executable(main main.cpp ${SOURCES_FILES})
target_sources(main PRIVATE ${SOURCE_FILES})

target_include_directories(main PRIVATE
  ${MLPACK_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE -static
  ${MLPACK_LIBRARIES}
)

