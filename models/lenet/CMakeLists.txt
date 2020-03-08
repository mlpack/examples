cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(lenet)

set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)

set(SOURCES
  lenet.hpp
  lenet_impl.hpp
)


if(DEBUG)
    message("Compilation with debug info (with ggdb3 flag)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
    message("Compilation without debug info (without ggdb3 flag)")
endif()

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)
