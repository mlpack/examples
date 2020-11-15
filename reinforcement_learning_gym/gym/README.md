# c++ client

## Dependencies

The c++ example agent has the following dependencies:

      Armadillo     >= 4.200.0
      Boost (system, thread)
      CMake         >= 2.8.5

## Installation

First, checkout the repository and change into the unpacked c++ client directory:

      git clone https://github.com/zoq/gym_tcp_api.git
      cd gym_tcp_api/cpp/

Then, make a build directory. The directory can have any name, not just 'build', but 'build' is sufficient.

      $ mkdir build
      $ cd build

The next step is to run CMake to configure the project. Running CMake is the equivalent to running ./configure with autotools

      $ cmake ../

Once CMake is configured, building the the example agent is as simple as typing 'make'.

      $ make

In a separate terminal, you can then run the example agent:

      $ ./example
