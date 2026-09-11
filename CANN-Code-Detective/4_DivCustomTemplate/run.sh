#!/bin/bash
set -e

# 获取当前脚本所在目录，确保在任何路径下执行都能找到正确文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " 1. Loading CANN Environment"
echo "=========================================="
# 加载 CANN 环境变量
if [ -n "$ASCEND_TOOLKIT_HOME" ] && [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    source $ASCEND_TOOLKIT_HOME/set_env.sh
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    source $ASCEND_HOME_PATH/set_env.sh
else
    echo "Error: Cannot find set_env.sh. Please check your CANN installation."
    exit 1
fi
echo "CANN Environment loaded successfully."

echo "=========================================="
echo " 2. Generating Operator Project (msopgen)"
echo "=========================================="
# 若算子工程不存在，则基于原型文件使用 msopgen 生成（工程化内容由参与者自行完成）
if [ -d "custom_op" ] && [ -f "custom_op/build.sh" ]; then
    echo ">>> custom_op already exists, skip msopgen generation."
else
    echo ">>> Generating operator project from div_custom_template.json..."
    msopgen gen -i div_custom_template.json -c ai_core-ascend910b1 -lan cpp -out ./custom_op
    echo ">>> Operator project generated at ./custom_op"
    echo ">>> 已基于910b1生成自定义算子工程 ./custom_op，如果芯片类型不一致或者有其他修改请手动生成相应工程"
    echo ">>> 请先在 custom_op/op_kernel 中完成算子核函数实现（参考 README.md），再重新执行 bash run.sh"
    exit 0
fi

echo "=========================================="
echo " 3. Building Custom Operator"
echo "=========================================="
cd custom_op
# 清理旧的构建目录以确保干净编译
rm -rf build_out
echo ">>> Running build.sh..."
bash build.sh

echo "=========================================="
echo " 4. Installing Custom Operator"
echo "=========================================="
RUN_FILE=$(ls build_out/custom_opp*.run 2>/dev/null | head -n 1)

if [ -z "$RUN_FILE" ]; then
    echo "Error: .run file not found in build_out. Build might have failed."
    exit 1
fi
echo ">>> Found installer: $RUN_FILE"
# 安装算子到用户目录
$RUN_FILE --install-path=${HOME}/
echo "Operator installed successfully."

echo "=========================================="
echo " 5. Loading Custom Operator Environment"
echo "=========================================="
# 【关键】必须 source 自定义算子的环境变量，否则运行时找不到算子库
if [ -f "${HOME}/vendors/customize/bin/set_env.bash" ]; then
    source ${HOME}/vendors/customize/bin/set_env.bash
    echo "Custom operator environment loaded."
else
    echo "Warning: Custom operator env script not found at ${HOME}/vendors/customize/bin/set_env.bash"
fi

echo "=========================================="
echo " 6. Building Test Case"
echo "=========================================="
cd "$SCRIPT_DIR"
# 编译测试代码
echo ">>> Compiling test/main.cpp..."
g++ -I$ASCEND_TOOLKIT_HOME/include \
    -I${HOME}/vendors/customize/op_api/include \
    -L$ASCEND_TOOLKIT_HOME/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lascendcl \
    -o execute_div_op
echo "Test case built successfully."

echo "=========================================="
echo " 7. Running Test Case"
echo "=========================================="
echo ">>> Executing..."
./execute_div_op
