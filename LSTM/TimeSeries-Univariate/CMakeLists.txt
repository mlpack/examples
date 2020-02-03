cmake_minimum_required(VERSION 3.1...3.16)

project(lstm-time-series)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SOURCES
  ${MODEL_SOURCE_DIR}/LSTMTimeSeriesUnivariate.cpp
  )

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
  message("Compilation without debug info (without ggdb3 flag)")
endif()


add_executable(LSTMTimeSeriesUnivariate ${SOURCES})
target_link_libraries(LSTMTimeSeriesUnivariate
    PRIVATE ${MLPACK_LIBRARIES}
    PRIVATE ${ARMADILLO_LIBRARIES}
    PRIVATE ${Boost_LIBRARIES})
target_include_directories(LSTMTimeSeriesUnivariate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../")



if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data)
  add_custom_command(TARGET LSTMTimeSeriesUnivariate POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../data)
endif()


if (NOT EXISTS ${PROJECT_BINARY_DIR}/../../../saved_models)
  add_custom_command(TARGET LSTMTimeSeriesUnivariate POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      ${PROJECT_BINARY_DIR}/../../saved_models)
endif()


if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data/electricity-usage.csv)
  add_custom_command(TARGET LSTMTimeSeriesUnivariate POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_SOURCE_DIR}/../../data/electricity-usage.csv
    ${PROJECT_BINARY_DIR}/../data/electricity-usage.csv)
endif()
