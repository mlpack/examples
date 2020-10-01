cmake_minimum_required (VERSION 2.8)
project(gym-tcp-api)

# Include modules in the CMake directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Set the CFLAGS and CXXFLAGS.
add_definitions("-Wno-deprecated-declarations")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fPIC -ffast-math ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "-std=c99 -O3 ${CMAKE_C_FLAGS} -lz")

# If using clang, we have to link against libc++ depending on the
# OS (at least on some systems). Further, gcc sometimes optimizes calls to
# math.h functions, making -lm unnecessary with gcc, but it may still be
# necessary with clang.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  IF (APPLE)
    # detect OS X version. Use '/usr/bin/sw_vers -productVersion' to
    # extract V from '10.V.x'.)
    EXEC_PROGRAM(/usr/bin/sw_vers ARGS
        -productVersion OUTPUT_VARIABLE MACOSX_VERSION_RAW)
    string(REGEX REPLACE
        "10\\.([0-9]+).*" "\\1"
        MACOSX_VERSION
        "${MACOSX_VERSION_RAW}")

     # OSX Lion (10.7) and OS X Mountain Lion (10.8) doesn't automatically
     # select the right stdlib.
    if(${MACOSX_VERSION} LESS 9)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
     endif(${MACOSX_VERSION} LESS 9)
  endif(APPLE)
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

# Find the libraries we need to compile against.
set(Boost_ADDITIONAL_VERSIONS
  "1.72.0" "1.72"
  "1.71.0" "1.71"
  "1.70.0" "1.70"
  "1.69.0" "1.69"
  "1.68.0" "1.68"
  "1.67.0" "1.67"
  "1.66.0" "1.66"
  "1.65.1" "1.65.0" "1.65"
  "1.64.1" "1.64.0" "1.64"
  "1.63.1" "1.63.0" "1.63"
  "1.62.1" "1.62.0" "1.62"
  "1.61.1" "1.61.0" "1.61"
  "1.60.1" "1.60.0" "1.60"
  "1.59.1" "1.59.0" "1.59"
  "1.58.1" "1.58.0" "1.58")

find_package(Boost 1.49 COMPONENTS system thread iostreams)

if (NOT Boost_FOUND)
  # Try again with non-multithreaded libraries ('mt' tag).
  set(Boost_USE_MULTITHREADED OFF)
  find_package(Boost 1.49 COMPONENTS system thread iostreams REQUIRED)
endif()

find_package(Armadillo 3.6.0 REQUIRED)

# Include directories for the dependencies.
include_directories(${CMAKE_SOURCE_DIR}/pjson)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})

MESSAGE(STATUS ${Boost_LIBRARIES})

# Set source file path.
set(gym_tcp_api_source
    example.cpp
    parser.hpp
    parser_impl.hpp
    space.hpp
    space_impl.hpp
    environment.hpp
    environment_impl.hpp
    client.hpp
    messages.hpp
    monitor.hpp
    monitor_impl.hpp
)

# Define the executable and link against the libraries we need to build the
# source.
add_executable(example ${gym_tcp_api_source})
target_link_libraries(example ${Boost_LIBRARIES} ${ARMADILLO_LIBRARIES})
