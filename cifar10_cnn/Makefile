TARGET := cifar_train
TARGET1 := cifar_eval
SRC := cifar_train.cpp
SRC1 := cifar_eval.cpp
LIBS_NAME := armadillo mlpack

CXX := g++
CXXFLAGS := -std=c++14 -Wall -g -rdynamic -Wextra -O3 -DNDEBUG -fopenmp

LDFLAGS := -fopenmp

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
