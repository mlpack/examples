cmake_minimum_required(VERSION 2.6)
project (augmented_lstm_baseline)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SOURCES
  ${MODEL_SOURCE_DIR}/lstm_baseline_main.cpp
)

add_executable(mlpack_lstm_baseline ${SOURCES})
target_link_libraries(mlpack_lstm_baseline ${MODELS_LIBRARIES})
