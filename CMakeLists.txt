# specify the minimum version. 
cmake_minimum_required(VERSION 3.1...3.16)

# set cmake policy.
if (${CMAKE_VERSION} VERSION_LESS 3.12 )
      cmake_policy( VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} )
endif()

# set the standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# specify the languages like C, CXX, CUDA.
project(models
        DESCRIPTION "model examples mlpack"
        LANGUAGES C CXX)
        
# Include modules in the CMake directory.
set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH}
   "${CMAKE_SOURCE_DIR}/CMake"
   "${CMAKE_SOURCE_DIR}/cmake/modules/")

# Now, find the libraries we need to compile against.  Several variables can be
# set to manually specify the directory in which each of these libraries
# resides.
#  ARMADILLO_LIBRARY - location of libarmadillo.so / armadillo.lib
#  ARMADILLO_INCLUDE_DIR - directory containing <armadillo>
#  ARMADILLO_INCLUDE_DIRS - directories necessary for Armadillo includes
#  MLPACK_INCLUDE_DIR - include directory for mlpack
#  MLPACK_LIBRARY - library directory for mlpack

find_package(Armadillo 8.400.0 REQUIRED)

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

# If we are on a Unix-like system, use the GNU install directories module.
# Otherwise set the values manually.
if (UNIX)
  include(GNUInstallDirs)
else ()
  set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
  set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
  set(CMAKE_INSTALL_MANDIR ${CMAKE_INSTALL_PREFIX}/man)
  set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_PREFIX}/share/doc/mlpack)
  set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)
endif ()

# This is as of yet unused.
option(PGO "Use profile-guided optimization if not a debug build" ON)

# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Ensure that we can't compile with clang 3.4, since this causes strange
  # issues.
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
    message(FATAL_ERROR "mlpack does not build correctly with clang < 3.5.  "
        "Please upgrade your compiler and reconfigure mlpack.")
  endif ()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ftemplate-depth=1000")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

  # To remove unused functions warnings.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
endif()
# These support libraries are used if we need to link against something
# specific.  This list is a subset of MLPACK_LIBRARIES.
set(COMPILER_SUPPORT_LIBRARIES "")

# If we are using MSVC, we need /bigobj.
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif ()

# If we are using MINGW, we need sections and big-obj, otherwise we create too
# many sections.
if(CMAKE_COMPILER_IS_GNUCC AND WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -Wa,-mbig-obj")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -Wa,-mbig-obj")
endif()

# If using clang, we have to link against libc++ depending on the
# OS (at least on some systems). Further, gcc sometimes optimizes calls to
# math.h functions, making -lm unnecessary with gcc, but it may still be
# necessary with clang.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (APPLE)
    # Detect OS X version. Use '/usr/bin/sw_vers -productVersion' to
    # extract V from '10.V.x'.
    exec_program(/usr/bin/sw_vers ARGS
        -productVersion OUTPUT_VARIABLE MACOSX_VERSION_RAW)
    string(REGEX REPLACE
        "10\\.([0-9]+).*" "\\1"
        MACOSX_VERSION
        "${MACOSX_VERSION_RAW}")

     # OSX Lion (10.7) and OS X Mountain Lion (10.8) doesn't automatically
     # select the right stdlib.
    if(${MACOSX_VERSION} LESS 9)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_SHARED_LINKER_FLAGS
          "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_MODULE_LINKER_FLAGS
          "${CMAKE_MODULE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
     endif()
  endif()

  # Link everything with -lm.
  set(COMPILER_SUPPORT_LIBRARIES ${COMPILER_SUPPORT_LIBRARIES} "m")
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} "m")
  # Use -pthread, but not on OS X.
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  endif ()
endif()

# If we're using gcc, then we need to link against pthreads to use std::thread,
# which we do in the tests.
if(CMAKE_COMPILER_IS_GNUCC)
  find_package(Threads)
  set(COMPILER_SUPPORT_LIBRARIES ${COMPILER_SUPPORT_LIBRARIES}
      ${CMAKE_THREAD_LIBS_INIT})
endif()

# Setup build for test coverage
if(BUILD_WITH_COVERAGE)
  # Currently coverage only works with GNU g++.
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # Find gcov and lcov
    find_program(GCOV gcov)
    find_program(LCOV lcov)

    if(NOT GCOV)
      message(FATAL_ERROR
          "gcov not found! gcov is required when BUILD_WITH_COVERAGE=ON.")
    endif()

    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} "supc++")
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} "quadmath")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline -fprofile-arcs -fkeep-inline-functions")
    message(STATUS "Adding debug compile options for code coverage.")
    # Remove optimizations for better line coverage
    set(DEBUG ON)

    if(LCOV)
      configure_file(CMake/mlpack_coverage.in mlpack_coverage @ONLY)
      add_custom_target(mlpack_coverage DEPENDS mlpack_test COMMAND ${PROJECT_BINARY_DIR}/mlpack_coverage)
    else()
      message(WARNING "'lcov' not found; local coverage report is disabled. "
          "Install 'lcov' and rerun cmake to generate local coverage report.")
    endif()
  else()
    message(FATAL_ERROR "BUILD_WITH_COVERAGE can only work with GNU environment.")
  endif()
endif()

# Debugging CFLAGS.  Turn optimizations off; turn debugging symbols on.
if(DEBUG)
  if (NOT MSVC)
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -ftemplate-backtrace-limit=0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -O0")
  endif()

  # mlpack uses it's own mlpack::backtrace class based on Binary File Descriptor
  # <bfd.h> and linux Dynamic Loader <libdl.h> and more portable version in future
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Bfd)
    find_package(LibDL)
    if(LIBBFD_FOUND AND LIBDL_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
      set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${LIBBFD_INCLUDE_DIRS}
          ${LIBDL_INCLUDE_DIRS})
      set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${LIBBFD_LIBRARIES}
          ${LIBDL_LIBRARIES})
      add_definitions(-DHAS_BFD_DL)
    else()
      message(WARNING "No libBFD and/or libDL has been found!")
    endif()
  endif()
else()
  add_definitions(-DARMA_NO_DEBUG)
  add_definitions(-DNDEBUG)
  if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O3")
  endif ()
endif()

# Profiling CFLAGS.  Turn profiling information on.
if(CMAKE_COMPILER_IS_GNUCC AND PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

# On Windows, Armadillo should be using LAPACK and BLAS but we still need to
# link against it.  We don't want to use the FindLAPACK or FindBLAS modules
# because then we are required to have a FORTRAN compiler (argh!) so we will try
# and find LAPACK and BLAS ourselves, using a slightly modified variant of the
# script Armadillo uses to find these.
if (WIN32)
  find_library(LAPACK_LIBRARY
      NAMES lapack liblapack lapack_win32_MT lapack_win32
      PATHS "C:/Program Files/Armadillo"
      PATH_SUFFIXES "examples/lib_win32/")

  if (NOT LAPACK_LIBRARY)
    message(FATAL_ERROR "Cannot find LAPACK library (.lib)!")
  endif ()

  find_library(BLAS_LIBRARY
      NAMES blas libblas blas_win32_MT blas_win32
      PATHS "C:/Program Files/Armadillo"
      PATH_SUFFIXES "examples/lib_win32/")

  if (NOT BLAS_LIBRARY)
    message(FATAL_ERROR "Cannot find BLAS library (.lib)!")
  endif ()

  # Piggyback LAPACK and BLAS linking into Armadillo link.
  set(ARMADILLO_LIBRARIES
       ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARY} ${LAPACK_LIBRARY})
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

# updated as time goes on and new versions are released.
set( BOOST_VERSION 1.72.0 )
# Settings for finding correct Boost libraries.
set( Boost_USE_STATIC_LIBS FALSE )
set( Boost_USE_MULTITHREADED TRUE )
set( Boost_USE_STATIC_RUNTIME FALSE )
set( Boost_COMPILER "-gcc8" )
set( Boost_ADDITIONAL_VERSIONS "${BOOST_VERSION}" )

find_package(Boost ${BOOST_VERSION}
      MODULE
      REQUIRED COMPONENTS program_options
                          unit_test_framework
                          serialization
                          graph)
# Make found targets globally available.
if ( Boost_FOUND )
  set_target_properties( Boost::boost
  Boost::program_options
  Boost::graph
  PROPERTIES IMPORTED_GLOBAL TRUE )
else()
  message(FATAL_ERROR "Could not found boost version ${BOOST_VERSION}")
endif ()

if (USE_OPENMP)
  find_package(OpenMP)
endif ()

if (OPENMP_FOUND)
  add_definitions(-DHAS_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
  # Disable warnings for all the unknown OpenMP pragmas.
  if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
  endif ()
  set(OpenMP_CXX_FLAGS "")
endif ()

# In Visual Studio, automatic linking is performed, so we don't need to worry
# about it.  Clear the list of libraries to link against and let Visual Studio
# handle it.
if (MSVC)
  link_directories(${Boost_LIBRARY_DIRS})
  set(Boost_LIBRARIES "")
  set(MODELS_LIBRARIES "")
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

enable_testing()
