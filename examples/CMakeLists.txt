cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(examples)

add_subdirectory(mnist_tutorial)
add_subdirectory(time_series_tutorial)


# Add directory name to sources.
set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)
