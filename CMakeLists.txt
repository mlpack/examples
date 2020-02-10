cmake_minimum_required(VERSION 2.6)
project(models C CXX)

# First, define all the compilation options.
# We default to debugging mode for developers.
option(FORCE_CXX11
    "Don't check that the compiler supports C++11, just assume it.  Make sure to specify any necessary flag to enable C++11 as part of CXXFLAGS."
    OFF)
option(DOWNLOAD_ENSMALLEN "If ensmallen is not found, download it." ON)
option(USE_OPENMP "If available, use OpenMP for parallelization." OFF)

# If we are not forcing C++11 support, check that the compiler supports C++11
# and enable it.
if (NOT FORCE_CXX11)
  include(CMake/CXX11.cmake)
  check_for_cxx11_compiler(HAS_CXX11)

  if(NOT HAS_CXX11)
    message(FATAL_ERROR "No C++11 compiler available!")
  endif()
  enable_cxx11()
endif ()

# Include modules in the CMake directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

# Set the CFLAGS and CXXFLAGS.
# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wextra -ftemplate-depth=1000")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif()

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

# Now, find the libraries we need to compile against.  Several variables can be
# set to manually specify the directory in which each of these libraries
# resides.
#   ARMADILLO_LIBRARY - location of libarmadillo.so / armadillo.lib
#   ARMADILLO_INCLUDE_DIR - directory containing <armadillo>
#   ARMADILLO_INCLUDE_DIRS - directories necessary for Armadillo includes
#   MLPACK_INCLUDE_DIR - include directory for mlpack
#   MLPACK_LIBRARY - library directory for mlpack

find_package(Armadillo 4.200.0 REQUIRED)

# If Armadillo was compiled without ARMA_64BIT_WORD and we are on a 64-bit
# system (where size_t will be 64 bits), suggest to the user that they should
# compile Armadillo with 64-bit words.  Note that with Armadillo 5.000.0 and
# newer, ARMA_64BIT_WORD is enabled by default.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  # Check the version, to see if ARMA_64BIT_WORD is enabled by default.
  set(ARMA_HAS_64BIT_WORD 0)
  if(NOT (${ARMADILLO_VERSION_MAJOR} LESS 5))
    set(ARMA_HAS_64BIT_WORD 1)
  else()
    # Can we open the configuration file?  If not, issue a warning.
    if(NOT EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
      message(WARNING "Armadillo configuration file "
          "(${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp) does not exist!")
    else()
      # We are on a 64-bit system.  Does Armadillo have ARMA_64BIT_WORD enabled?
      file(READ "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp" ARMA_CONFIG)
      string(REGEX MATCH
          "[\r\n][ ]*#define ARMA_64BIT_WORD"
          ARMA_HAS_64BIT_WORD_PRE
          "${ARMA_CONFIG}")

      string(LENGTH "${ARMA_HAS_64BIT_WORD_PRE}" ARMA_HAS_64BIT_WORD)
    endif()
  endif()

  if(ARMA_HAS_64BIT_WORD EQUAL 0)
    message(WARNING "This is a 64-bit system, but Armadillo was compiled "
        "without 64-bit index support.  Consider recompiling Armadillo with "
        "ARMA_64BIT_WORD to enable 64-bit indices (large matrix support). "
        "mlpack will still work without ARMA_64BIT_WORD defined, but will not "
        "scale to matrices with more than 4 billion elements.")
  endif()
else()
  # If we are on a 32-bit system, we must manually specify the size of the word
  # to be 32 bits, since otherwise Armadillo will produce a warning that it is
  # disabling 64-bit support.
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    add_definitions(-DARMA_32BIT_WORD)
  endif ()
endif()

# On Windows, Armadillo should be using LAPACK and BLAS but we still need to
# link against it.  We don't want to use the FindLAPACK or FindBLAS modules
# because then we are required to have a FORTRAN compiler (argh!) so we will try
# and find LAPACK and BLAS ourselves, using a slightly modified variant of the
# script Armadillo uses to find these.
if (WIN32)
  ADD_DEFINITIONS(/bigobj)
  find_library(LAPACK_LIBRARY
      NAMES lapack liblapack lapack_win32_MT lapack_win32
      PATHS "C:/Program Files/Armadillo"
      PATH_SUFFIXES "examples/lib_win32/")

  if (NOT LAPACK_LIBRARY)
    message(FATAL_ERROR "Cannot find LAPACK library (.lib)!")
  endif ()
  
  find_package(BLAS)
  find_package(LAPACK)

  if (NOT BLAS_LIBRARIES)
    message(FATAL_ERROR "Cannot find BLAS library (.lib)!")
  endif ()

  # Piggyback LAPACK and BLAS linking into Armadillo link.
  set(ARMADILLO_LIBRARIES
       ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif ()

find_package(MLPACK REQUIRED)
find_package(Ensmallen 1.10.0)
if (NOT ENSMALLEN_FOUND)
  if (DOWNLOAD_ENSMALLEN)
    file(DOWNLOAD http://www.ensmallen.org/files/ensmallen-latest.tar.gz
        "${CMAKE_BINARY_DIR}/deps/ensmallen-latest.tar.gz"
        STATUS ENS_DOWNLOAD_STATUS_LIST LOG ENS_DOWBLOAD_LOG
        SHOW_PROGRESS)
    list(GET ENS_DOWNLOAD_STATUS_LIST 0 ENS_DOWNLOAD_STATUS)
    if (ENS_DOWNLOAD_STATUS EQUAL 0)
      execute_process(COMMAND ${CMAKE_COMMAND} -E
          tar xzf "${CMAKE_BINARY_DIR}/deps/ensmallen-latest.tar.gz"
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/deps/")

      # Get the name of the directory.
      file (GLOB ENS_DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
          "${CMAKE_BINARY_DIR}/deps/ensmallen-[0-9]*.[0-9]*.[0-9]*")
      list(LENGTH ENS_DIRECTORIES ENS_DIRECTORIES_LEN)
      if (ENS_DIRECTORIES_LEN EQUAL 1)
        list(GET ENS_DIRECTORIES 0 ENSMALLEN_INCLUDE_DIR)
        set(ENSMALLEN_INCLUDE_DIR
             "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include")
        message(STATUS
            "Successfully downloaded ensmallen into ${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/")
      else ()
        message(FATAL_ERROR "Problem unpacking ensmallen!  Expected only one directory ensmallen-x.y.z/; found ${ENS_DIRECTORIES}. Try removing the directory ${CMAKE_BINARY_DIR}/deps and reconfiguring.")
      endif ()
    else ()
      list(GET ENS_DOWNLOAD_STATUS_LIST 1 ENS_DOWNLOAD_ERROR)
      message(FATAL_ERROR
          "Could not download ensmallen! Error code ${ENS_DOWNLOAD_STATUS}: ${ENS_DOWNLOAD_ERROR}!  Error log: ${ENS_DOWBLOAD_LOG}")
    endif ()
  else ()
    # Release versions will have ensmallen packaged with the release so we can
    # just reference that.
    if (EXISTS "${CMAKE_SOURCE_DIR}/src/mlpack/core/optimizers/ensmallen/ensmallen.hpp")
      set(ENSMALLEN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/mlpack/core/optimizers/ensmallen")
    else ()
      message(FATAL_ERROR
          "Cannot find ensmallen headers!  Try setting ENSMALLEN_INCLUDE_DIR!")
    endif ()
  endif ()
endif ()

if (USE_OPENMP)
  message(WARNING "Only use OpenMP if mlpack has been compiled with OpenMP.")
  find_package(OpenMP)
endif ()

if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Unfortunately this configuration variable is necessary and will need to be
# updated as time goes on and new versions are released.
set(Boost_ADDITIONAL_VERSIONS
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
  "1.58.1" "1.58.0" "1.58"
  "1.57.1" "1.57.0" "1.57"
  "1.56.1" "1.56.0" "1.56"
  "1.55.1" "1.55.0" "1.55"
  "1.54.1" "1.54.0" "1.54"
  "1.53.1" "1.53.0" "1.53"
  "1.52.1" "1.52.0" "1.52"
  "1.51.1" "1.51.0" "1.51"
  "1.50.1" "1.50.0" "1.50"
  "1.49.1" "1.49.0" "1.49")
# Disable forced config-mode CMake search for Boost, which only imports targets
# and does not set the variables that we need.
#
# TODO for the brave: transition all mlpack's CMake to 'target-based modern
# CMake'.  Good luck!  You'll need it.
set(Boost_NO_BOOST_CMAKE 1)
find_package(Boost 1.49
    COMPONENTS
      program_options
      unit_test_framework
      serialization
    REQUIRED
)

link_directories(${Boost_LIBRARY_DIRS})

# In Visual Studio, automatic linking is performed, so we don't need to worry
# about it.  Clear the list of libraries to link against and let Visual Studio
# handle it.
if (MSVC)
  link_directories(${Boost_LIBRARY_DIRS})
  set(CMAKE_MSVCIDE_RUN_PATH ${CMAKE_MSVCIDE_RUN_PATH} ${Boost_LIBRARY_DIRS})
  message("boost lib dirs ${Boost_LIBRARY_DIRS}")
  set(Boost_LIBRARIES "")
endif ()

# In Visual Studio, automatic linking is performed, so we don't need to worry
# about it.  Clear the list of libraries to link against and let Visual Studio
# handle it.
if (MSVC)
  link_directories(${Boost_LIBRARY_DIRS})
  set(Boost_LIBRARIES "")
endif ()

link_directories(${Boost_LIBRARY_DIRS})

# Include directories for the previous dependencies.
set(MODELS_INCLUDE_DIRS ${MLPACK_INCLUDE_DIR}
                        ${ENSMALLEN_INCLUDE_DIR}
                        ${ARMADILLO_INCLUDE_DIR}
                        ${Boost_INCLUDE_DIRS})
set(MODELS_LIBRARIES ${MLPACK_LIBRARY}
                     ${ARMADILLO_LIBRARIES}
                     ${Boost_LIBRARIES})
set(MODELS_LIBRARY_DIRS ${MODELS_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})

include_directories(${MODELS_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# Recurse into each model.
set(DIRS
  # Model directories.
  Kaggle/DigitRecognizer
  Kaggle/DigitRecognizerCNN
  Kaggle/DigitRecognizerBatchNorm
  vae/vae_mnist
  vae/vae_cnn_mnist
  LSTM/TimeSeries-Univariate
  LSTM/TimeSeries-Multivariate
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()
