BIN_PATH := .
OBJ_PATH := obj
SRC_PATH := .

TARGET_NAME := mnist_simple
TARGET := $(BIN_PATH)/$(TARGET_NAME)
MAIN_SRC := mnist_simple.cpp
LIBS_NAME := armadillo mlpack boost_program_options boost_serialization

CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra -fopenmp
LDFLAGS  := -fopenmp
DBGFLAG := -g

SRCS := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cpp)))
OBJS := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRCS)))))
LIBS := $(addprefix -l,$(LIBS_NAME))

DISTCLEAN_LIST := $(OBJS)

CLEAN_LIST := $(TARGET) \
			  $(DISTCLEAN_LIST)

# default rule
default: all

#Linking
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LIBS) $(LDFLAGS) -o $@

#Compiling
$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CXX) $(CXXFLAGS) -MP -MMD -c $< -o $@ 

# phony rules
.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)
