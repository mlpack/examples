cmake_minimum_required(VERSION 2.6)
project(kaggle-digit-recognizer)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES ${MODEL_SOURCE_DIR}/DigitRecognizer.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")


add_executable(DigitRecognizer ${SOURCES})
target_link_libraries(DigitRecognizer ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
