cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(mnist_tutorial)

option(DEBUG "DEBUG" OFF)
option(DOWNLOAD_MNIST "DOWNLOAD_MNIST" OFF)
# Download all datasets used in examples.
option(DOWNLOAD_DATASETS "DOWNLOAD_DATASETS" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  mnist_tutorial_cnn.cpp
  mnist_tutorial_nn.cpp
)

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
   message("Compilation without debug info (without ggdb3 flag)")
endif()

foreach(file ${SOURCES})
  string( REPLACE ".cpp" "" name ${file})
  add_executable(${name} ${MODEL_SOURCE_DIR}/${file})
  target_link_libraries(${name}
      ${MLPACK_LIBRARIES}
      ${ARMADILLO_LIBRARIES}
      ${Boost_LIBRARIES})
endforeach()

function(download_file url filename)

  if(NOT EXISTS ${filename})
    message("Downloading "  ${filename})
    file(DOWNLOAD ${url} ${filename}
         SHOW_PROGRESS
         TIMEOUT 60)  # seconds
  endif()

endfunction(download_file)

if(DOWNLOAD_MNIST OR DOWNLOAD_DATASETS)
download_file(
    https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_train.csv
    ./../data/mnist_train.csv)

download_file(
    https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_test.csv
    ./../data/mnist_test.csv)
endif()