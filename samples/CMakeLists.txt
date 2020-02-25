cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(model-zoo-samples)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")

set(SOURCES
  ${MODEL_SOURCE_DIR}/object_classification.cpp
  )

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()

add_executable(ModelZooSamples ${SOURCES})
target_link_libraries(ModelZooSamples
  ${MLPACK_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${Boost_LIBRARIES})

if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data)
  add_custom_command(TARGET ModelZooSamples POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../data)
endif()
