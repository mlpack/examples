# This is a simple Makefile used to build the source code.
# If you're not using the Armadillo wrapper, replace `armadillo` with linker commands
# for the BLAS and LAPACK libraries that you are using.

TARGET := denoiseNet_load_weights
SRC := denoiseNet_load_weights.cpp
LIBS_NAME := armadillo mlpack boost_serialization

CXX := g++
CXXFLAGS += -std=c++11 -Wall -Wextra -O3 -DNDEBUG -fopenmp
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
# CXXFLAGS += -std=c++11 -Wall -Wextra -g -O0
LDFLAGS += -L . # /path/to/mlpack/library/  # if installed locally.
# Add header directories for any includes that aren't on the
# default compiler search path.
INCLFLAGS := -I .
CXXFLAGS += $(INCLFLAGS)

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l,$(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS)

# default rule
default: all

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)
