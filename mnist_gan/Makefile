
TARGET := mnist_gan
SRC := mnist_gan.cpp
LIBS_NAME := armadillo mlpack

CXX := g++
CXXFLAGS += -std=c++11 -Wall -Wextra -O3 -DNDEBUG
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
# CXXFLAGS += -std=c++11 -Wall -Wextra -g -O0
LDFLAGS  += -fopenmp
LDFLAGS += -lboost_serialization
LDFLAGS += -larmadillo
LDFLAGS += -L /home/viole/mlpack/build/lib/ # /path/to/mlpack/library/  # if installed locally.
# Add header directories for any includes that aren't on the
# default compiler search path.
INCLFLAGS := -I /home/viole/mlpac/build/include/
CXXFLAGS  += $(INCLFLAGS)

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