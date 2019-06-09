cmake_minimum_required(VERSION 2.6)
project(vgg19-mnist)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  ${MODEL_SOURCE_DIR}/model.cpp
  ${MODEL_SOURCE_DIR}/VGG19_impl.hpp
  ${MODEL_SOURCE_DIR}/VGG19.hpp
  )

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()

add_executable(vgg19 ${SOURCES})
target_link_libraries(vgg19
  ${MLPACK_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${Boost_LIBRARIES})

if (NOT EXISTS ${PROJECT_BINARY_DIR}/../saved_models)
  add_custom_command(TARGET vgg19 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../saved_models)
endif()

## Avoiding uncessary copy
#if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data)
#  add_custom_command(TARGET vgg19 POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_CURRENT_SOURCE_DIR}/../../Kaggle/kaggle_train_test_dataset.zip
#    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/../)
#endif()

