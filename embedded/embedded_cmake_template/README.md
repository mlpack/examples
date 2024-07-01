## CMake files for mlpack

This is a set of CMakefiles that autodowload all the mlpack dependencies. The
objective of this example is to provide the minimum configurations that allow
to build an mlpack application with in the objective of embedded system.

These files contains the necessary cmake files that are required in the
embedded context, especially the cross compilation step, and to build a
statically linked binary, you can use these configs to build any mlpack
application for any platform as long as you modify the configs accordingly.

The main CMakeLists.txt assumes that the project name is `main` and that you
have a file that is called `main.cpp` that contains the mlpack C++ code you are
trying to compile. Feel free to modify these the way it suits you.

If you are looking to integrate these CMakeFiles into your project, feel free
to modify them the way you want. The point of these files to provide a simple
working example and to keep things simple
