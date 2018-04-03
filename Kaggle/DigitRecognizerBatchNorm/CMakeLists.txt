cmake_minimum_required(VERSION 2.6)
project(kaggle-digit-recognizer)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES ${MODEL_SOURCE_DIR}/DigitRecognizerBatchNorm.cpp)

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()

add_executable(DigitRecognizerBatchNorm ${SOURCES})
target_link_libraries(DigitRecognizerBatchNorm
  ${MLPACK_LIBRARIES} 
  ${ARMADILLO_LIBRARIES})

add_custom_command(TARGET DigitRecognizerBatchNorm POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    kaggle-digit-recognizer_SOURCE_DIR/data/
    ${PROJECT_BINARY_DIR}/../..)
