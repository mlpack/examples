cmake_minimum_required(VERSION 2.6)
project(lstm-sentiment-analysis)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  ${MODEL_SOURCE_DIR}/LSTMSentimentAnalysis.cpp
  )

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()


add_executable(LSTMSentimentAnalysis ${SOURCES})
target_link_libraries(LSTMSentimentAnalysis
  ${MLPACK_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${Boost_LIBRARIES})



if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data)
  add_custom_command(TARGET LSTMSentimentAnalysis POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../data)
endif()


if (NOT EXISTS ${PROJECT_BINARY_DIR}/../../../saved_models)
  add_custom_command(TARGET LSTMSentimentAnalysis POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../../saved_models)
endif()


if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data/data_1000.txt)
  file(DOWNLOAD https://transfer.sh/wTUKg/data_1000.txt
          "${PROJECT_BINARY_DIR}/../data/data_1000.txt"
          STATUS IMDB_DATA_STATUS_LIST
          LOG IMDB_DATA_LOG
          SHOW_PROGRESS)
endif()


