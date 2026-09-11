
# shared library name
LIBRARY_NAME := ascend_all_ops

# src file list
SRCS := /workspace/code/op_host/relu_def.cpp /workspace/code/op_host/relu_infershape.cpp /workspace/code/op_host/relu_tiling.cpp

# output dir
BUILD_DIR := /workspace/code/build/autogen
LIB_DIR := $(BUILD_DIR)/

# target objs
OBJS_CPP := $(SRCS:%.cpp=$(BUILD_DIR)/$(LIBRARY_NAME)/%.o)
OBJS := $(OBJS_CPP:%.cc=$(BUILD_DIR)/$(LIBRARY_NAME)/%.o)

CXX := /usr/bin/c++
CXXFLAGS := -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/include -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/asc/include/tiling -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc/op_common -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc/base -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc/exe_graph -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc/graph -I/usr/local/Ascend/cann-9.1.0/include
LDFLAGS := -shared

TARGET := $(LIB_DIR)/lib$(LIBRARY_NAME).so

all: $(TARGET)
	rm -rf $(BUILD_DIR)/$(LIBRARY_NAME)/

$(shell mkdir -p $(dir $(OBJS)) $(LIB_DIR))

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^  -lexe_graph -lregister -ltiling_api -L/usr/local/Ascend/cann-9.1.0/lib64

$(BUILD_DIR)/$(LIBRARY_NAME)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/$(LIBRARY_NAME)/%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
