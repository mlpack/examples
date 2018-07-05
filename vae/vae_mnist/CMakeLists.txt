cmake_minimum_required(VERSION 2.6)
project(vae-vae-mnist)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")

set(SOURCES
  ${MODEL_SOURCE_DIR}/vae.cpp
  )

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()

add_executable(vae ${SOURCES})
target_link_libraries(vae
  ${MLPACK_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${Boost_Serialization_LIBRARIES})

add_custom_command(TARGET vae POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../vae/data/
    ${PROJECT_BINARY_DIR}/../..)
