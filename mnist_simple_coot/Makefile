# This is a simple Makefile used to build the example source code.
# This example might requires some modifications in order to work correctly on
# your system.
# This example trains mlpack neural network on GPU via OpenCL or CUDA, and it uses
# the bandicoot library.

TARGET := mnist_simple_coot
SRC := mnist_simple_coot.cpp
LIBS_NAME := bandicoot

CXX := g++
CXXFLAGS += -std=c++14 -Wall -Wextra -O3 -DNDEBUG -fopenmp
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
# CXXFLAGS += -std=c++14 -Wall -Wextra -g -O0

LDFLAGS  += -fopenmp
# Add header directories for any includes that aren't on the
# default compiler search path.
INCLFLAGS := -I .
# If you have mlpack or ensmallen installed somewhere nonstandard, uncomment and
# update the lines below.
INCLFLAGS += -I/opt/cuda/targets/x86_64-linux/include
# INCLFLAGS += -I/path/to/ensmallen/include/
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
