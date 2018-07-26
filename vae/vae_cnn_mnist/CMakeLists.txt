cmake_minimum_required(VERSION 2.6)
project(vae-cnn-mnist)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  ${MODEL_SOURCE_DIR}/vae_cnn.cpp
  )

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()

add_executable(vae_cnn ${SOURCES})
target_link_libraries(vae_cnn
  ${MLPACK_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${Boost_LIBRARIES})

if (NOT EXISTS ${PROJECT_BINARY_DIR}/../mnist_full.csv)
  add_custom_command(TARGET vae_cnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E tar xvzf
      ${CMAKE_CURRENT_SOURCE_DIR}/../../data/mnist_full.tar.xz
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/../)
endif()

if (NOT EXISTS ${PROJECT_BINARY_DIR}/../saved_models)
  add_custom_command(TARGET vae_cnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../saved_models)
endif()
