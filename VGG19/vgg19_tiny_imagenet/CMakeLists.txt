cmake_minimum_required(VERSION 2.6)
project(vgg19-tiny-imagenet)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  ${MODEL_SOURCE_DIR}/model.cpp
  ${MODEL_SOURCE_DIR}/dataloader.hpp
  ${MODEL_SOURCE_DIR}/../../VGG19_impl.hpp
  ${MODEL_SOURCE_DIR}/../../VGG19.hpp
  )

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_STB -lboost_system -lboost_filesystem")
add_executable(VGG19Imagenet ${SOURCES})
target_link_libraries(VGG19Imagenet
  ${MLPACK_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${Boost_LIBRARIES})

if (NOT EXISTS ${PROJECT_BINARY_DIR}/../../saved_models)
  add_custom_command(TARGET VGG19Imagenet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../../saved_models)
endif()

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    imagenet
    PREFIX ${PROJECT_BINARY_DIR}/../../imagenet
    GIT_REPOSITORY https://github.com/seshuad/IMagenet.git
    TIMEOUT 10
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)
