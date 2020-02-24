cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(model-zoo-samples)

option(DEBUG "DEBUG" OFF)

set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)

add_executable(model_zoo_samples
   object_classification.cpp
)

target_link_libraries(ModelZooSamples
  models
  PRIVATE ${MLPACK_LIBRARIES}
  PRIVATE ${ARMADILLO_LIBRARIES})

target_include_directories(ModelZooSamples PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../")