TARGET := cifar_train
TARGET1 := cifar_eval
SRC := cifar_train.cpp
SRC1 := cifar_eval.cpp
LIBS_NAME := armadillo

CXX := g++
CXXFLAGS := -std=c++17 -Wall -g -rdynamic -Wextra -O3 -DNDEBUG -fopenmp

LDFLAGS := -fopenmp

INCLFLAGS := -I .
# If you have mlpack or ensmallen installed somewhere nonstandard, uncomment and
# update the lines below.
# INCLFLAGS += -I/path/to/mlpack/include/
# INCLFLAGS += -I/path/to/ensmallen/include/
CXXFLAGS += $(INCLFLAGS)

OBJS := $(SRC:.cpp=.o)
OBJS1 := $(SRC1:.cpp=.o)
LIBS := $(addprefix -l, $(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS) $(TARGET1) $(OBJS1)

default: all

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

$(TARGET1): $(OBJS1)
	$(CXX) $(CXXFLAGS) $(OBJS1) -o $(TARGET1) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET) $(TARGET1)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -rf $(CLEAN_LIST)
