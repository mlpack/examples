# This is a simple Makefile used to build the example source code.
# This example might requires some modifications in order to work correctly on
# your system.
# If you're not using the Armadillo wrapper, replace `armadillo` with linker commands
# for the BLAS and LAPACK libraries that you are using.

TARGET := mnist_simple
SRC := mnist_simple.cpp
LIBS_NAME := armadillo mlpack boost_serialization

CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra
LDFLAGS  := -fopenmp
DBGFLAG := -g
INCLFLAGS := -I . # Add header directories if there is any.

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l,$(LIBS_NAME))
CLEAN_LIST := $(TARGET)

# default rule
default: all

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLFLAGS) $(LIBS) $(LDFLAGS) $(OBJS) -o $(TARGET)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)
