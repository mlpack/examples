cmake_minimum_required(VERSION 3.11)
project(main)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
include(CMake/Autodownload.cmake)
include(CMake/ConfigureCrossCompile.cmake)
include(CMake/Findmlpack.cmake)

option(USE_OPENMP "If available, use OpenMP for parallelization." ON)
# Set required standard to C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If we're using gcc, then we need to link against pthreads to use std::thread,
# which we do in the tests.
if (CMAKE_COMPILER_IS_GNUCC)
  find_package(Threads)
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

find_package(mlpack PATHS ${CMAKE_BINARY_DIR})
if (NOT MLPACK_FOUND)
  get_deps(https://github.com/mlpack/mlpack/archive/refs/tags/4.5.1.tar.gz mlpack 4.5.1.tar.gz)
  set(MLPACK_INCLUDE_DIRS ${GENERIC_INCLUDE_DIR})
  find_package(mlpack REQUIRED)
endif()

search_openblas(0.3.26)

find_package(Armadillo PATHS ${CMAKE_BINARY_DIR})
if (NOT ARMADILLO_FOUND)
  get_deps(https://files.mlpack.org/armadillo-12.6.5.tar.gz armadillo armadillo-12.6.5.tar.gz)
  set(ARMADILLO_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
endif()
# Include directories for the previous dependencies.
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})

# Find stb_image.h and stb_image_write.h.
get_deps(https://mlpack.org/files/stb.tar.gz stb stb.tar.gz)
set(STB_IMAGE_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${STB_IMAGE_INCLUDE_DIR}")

# Find ensmallen.
get_deps(https://www.ensmallen.org/files/ensmallen-latest.tar.gz ensmallen ensmallen-latest.tar.gz)
set(ENSMALLEN_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${ENSMALLEN_INCLUDE_DIR}")

# Find cereal.
get_deps(https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.0.tar.gz cereal cereal-1.3.0.tar.gz)
set(CEREAL_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIR})

# Detect OpenMP support in a compiler. If the compiler supports OpenMP, flags
# to compile with OpenMP are returned and added.  Note that MSVC does not
# support a new-enough version of OpenMP to be useful.
if (USE_OPENMP)
  find_package(OpenMP)
endif ()

if (OpenMP_FOUND AND OpenMP_CXX_VERSION VERSION_GREATER_EQUAL 3.0.0)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
else ()
  # Disable warnings for all the unknown OpenMP pragmas.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set(OpenMP_CXX_FLAGS "")
endif ()

include_directories(BEFORE ${MLPACK_INCLUDE_DIRS})
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src/)

# Finally, add any cross-compilation support libraries (they may need to come
# last).  If we are not cross-compiling, no changes will happen here.
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CROSS_COMPILE_SUPPORT_LIBRARIES})

add_executable(main main.cpp ${SOURCES_FILES})
target_sources(main PRIVATE ${SOURCE_FILES})

target_include_directories(main PRIVATE
  ${MLPACK_INCLUDE_DIRS}
  /meta/mlpack/src/
)

target_link_libraries(main PRIVATE -static
  ${MLPACK_LIBRARIES}
)

