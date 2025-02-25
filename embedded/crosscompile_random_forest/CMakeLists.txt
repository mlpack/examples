cmake_minimum_required(VERSION 3.6)
project(RandomForest)

include(CMake/ConfigureCrossCompile.cmake)
include(CMake/mlpack.cmake)

option(USE_OPENMP "If available, use OpenMP for parallelization." ON)

fetch_mlpack(ON)

add_executable(RandomForest main.cpp ${SOURCES_FILES})
target_sources(RandomForest PRIVATE ${SOURCE_FILES})

target_include_directories(RandomForest PRIVATE
  ${MLPACK_INCLUDE_DIRS}
)

target_link_libraries(RandomForest PRIVATE -static
  ${MLPACK_LIBRARIES}
)

