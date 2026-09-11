#!/bin/bash
# 激活cann环境，可根据实际情况修改，在线环境一般不用修改
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cd build/ && \
cmake .. && \
make && \
./mul_test