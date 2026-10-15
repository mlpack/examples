# Build for the movement-recognition example relevant to the movement
# recognition tutorial,
#
#   imu_test  -- sensor check + magnetometer calibration
#   collect   -- record sensor data to CSV
#   train     -- train a neural network from the CSVs
#   infer     -- live inference from the IMU
#
# Host build (your computer):
#   cmake .. && make
#
# Cross-compile for the Milk-V Duo with Bootlin riscv64-lp64d musl toolchain as
# follows:
#   cmake -DCMAKE_CROSSCOMPILING=ON -DARCH_NAME=RV64GCV \
#         -DCMAKE_TOOLCHAIN_FILE=../CMake/crosscompile-toolchain.cmake \
#         -DTOOLCHAIN_PREFIX=/path/to/.../bin/riscv64-buildroot-linux-musl- \
#         -DCMAKE_SYSROOT=/path/to/.../riscv64-buildroot-linux-musl/sysroot \
#         -DOPENBLAS_PATCHES=../CMake/patches/openblas-riscv64-low-memory.patch ..
#   make
#
# Build a single program with, e.g., `make collect` or `make train`.

cmake_minimum_required(VERSION 3.11)
project(movement_recognition CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CMake/ConfigureCrossCompile.cmake)
include(CMake/mlpack.cmake)

set(MLPACK_DISABLE_OPENMP ON)
set(MLPACK_DISABLE_STB ON)

# Download + configure mlpack and its dependencies,
# OpenBLAS, built single-threaded -- see ConfigureCrossCompile.cmake).
fetch_mlpack(ON)

if(CMAKE_CROSSCOMPILING)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
  file(GLOB OPENBLAS_LIB "${CMAKE_BINARY_DIR}/deps/OpenBLAS-*/libopenblas.a")
  set(MLPACK_EXTRA_LIBS ${OPENBLAS_LIB} gfortran m)
endif()

# The GY-89 driver is self-contained: the I2C wrapper and the three chip drivers.
# This driver is needed to collect the data from the sensor.
set(DRIVER_SOURCES
  driver/imu.cpp
  driver/l3gd20h.cpp
  driver/lsm303d.cpp
  driver/bmp180.cpp
  driver/i2c_bus.cpp
)

add_executable(imu_test driver/imu_test.cpp ${DRIVER_SOURCES})
add_executable(collect  collect/collect.cpp ${DRIVER_SOURCES})

foreach(tool imu_test collect)
  target_compile_options(${tool} PRIVATE -Os -fno-exceptions -fno-rtti -ffunction-sections -fdata-sections)
  target_link_libraries(${tool} PRIVATE -Wl,--gc-sections pthread)
endforeach()

add_executable(train train/train.cpp)
target_include_directories(train SYSTEM PRIVATE ${MLPACK_INCLUDE_DIRS})
target_link_libraries(train PRIVATE
  ${MLPACK_LIBRARIES}
  ${MLPACK_EXTRA_LIBS}
  pthread
)

add_executable(infer infer/infer.cpp ${DRIVER_SOURCES})
target_include_directories(infer SYSTEM PRIVATE ${MLPACK_INCLUDE_DIRS})
target_link_libraries(infer PRIVATE
  ${MLPACK_LIBRARIES}
  ${MLPACK_EXTRA_LIBS}
  pthread
)
