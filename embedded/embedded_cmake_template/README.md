## CMake files to cross compile an mlpack application

This directory contains a minimal and simple CMake configuration that
autodownloads all the mlpack dependencies.  This example can be used as a simple
starting point for building an mlpack application that will be compiled to an
embedded system.

This example contains the necessary CMake files that are required in the
embedded context, especially for cross-compilation, and to build a statically
linked binary.  You can use this configuration to build any mlpack
application for any platform, as long as you modify the configuration
accordingly.

The main CMakeLists.txt assumes that the project name is `main` and that you
have a file that is called `main.cpp` that contains the mlpack C++ code you are
trying to compile. Feel free to modify these the way it suits you.

Please refer to our documentation for an entire guide on cross compilation.
Usually, we use buildroot toolchain to achieve so, To crosscompile, please 
modify the following command accordingly:

```
cmake -DBUILD_TESTS=ON -DBOARD_NAME="RPI2" -DCMAKE_CROSSCOMPILE=ON -DCMAKE_TOOLCHAIN_FILE=/path/to/mlpack/board/crosscompile-toolchain.cmake -DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2023.08-1/bin/arm-buildroot-linux-gnueabihf-  -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot  ../
```
