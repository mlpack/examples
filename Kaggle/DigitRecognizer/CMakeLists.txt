cmake_minimum_required(VERSION 2.6)
project(kaggle-digit-recognizer)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCESFFN ${MODEL_SOURCE_DIR}/FFN/FFN.cpp)
set(SOURCESCNN ${MODEL_SOURCE_DIR}/CNN/CNN.cpp)

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()

add_executable(DigitRecognizerFFN ${SOURCESFFN})
add_executable(DigitRecognizerCNN ${SOURCESCNN})
target_link_libraries(DigitRecognizerFFN 
  ${MLPACK_LIBRARIES} 
  ${ARMADILLO_LIBRARIES})
target_link_libraries(DigitRecognizerCNN 
  ${MLPACK_LIBRARIES} 
  ${ARMADILLO_LIBRARIES})

add_custom_command(TARGET DigitRecognizerFFN POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ 
${PROJECT_BINARY_DIR}/../..)

add_custom_command(TARGET DigitRecognizerCNN POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ 
${PROJECT_BINARY_DIR}/../..)
