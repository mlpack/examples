TARGET := cifar_train
SRC := cifar_train.cpp
LIBS_NAME := armadillo mlpack

CXX := g++
CXXFLAGS := -std=c++11 -Wall -g -rdynamic -Wextra -O3 -DNDEBUG -fopenmp

LDFLAGS := -fopenmp

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l, $(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS)

default: all

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -rf $(CLEAN_LIST)
