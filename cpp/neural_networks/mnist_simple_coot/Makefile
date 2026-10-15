# This is a simple Makefile used to build the example source code.
# This example might requires some modifications in order to work correctly on
# your system.
# This example trains mlpack neural network on GPU via OpenCL, CUDA or Vulkan, and it uses
# the bandicoot library.

TARGET := mnist_simple_coot
SRC := mnist_simple_coot.cpp
#SRC := main.cpp
LIBS_NAME := bandicoot openblas armadillo

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -O3 -DNDEBUG -fopenmp
# Select the bandicoot backend to run the example on.
CXXFLAGS += -DCOOT_DEFAULT_BACKEND=CUDA_BACKEND
# CXXFLAGS += -DCOOT_DEFAULT_BACKEND=CL_BACKEND
# CXXFLAGS += -DCOOT_DEFAULT_BACKEND=VULKAN_BACKEND
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
# CXXFLAGS += -std=c++17 -Wall -Wextra -g -O0

LDFLAGS  += -fopenmp
# Add header directories for any includes that aren't on the
# default compiler search path.

INCLFLAGS := -I .
# If you have mlpack, ensmallen or bandicoot installed somewhere nonstandard,
# uncomment and update the lines below.
# INCLFLAGS += -I/path/to/mlpack/include/
# INCLFLAGS += -I/path/to/ensmallen/include/
# INCLFLAGS += -I/path/to/bandicoot/include/
# The CUDA backend of bandicoot needs the CUDA headers; comment this out if
# bandicoot was built without CUDA support.
INCLFLAGS += -I/usr/local/cuda/include
CXXFLAGS += $(INCLFLAGS)
CXXFLAGS += $(INCLFLAGS)

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l,$(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS)

# default rule
default: all

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)
