# PR 信息

## 标题

```
【代码侦探Challenge04】 JeffDing div_custom_template
```

## 内容

```markdown
本 PR 完成 Challenge04-DivCustomTemplate 工程化算子开发，实现两个输入 Tensor 的逐元素除法：z = x / y

整体基于 `msopgen` 生成的 Ascend C 自定义算子工程进行补充开发，并完成 Host、Kernel、算子打包安装以及 ACLNN 调用验证。

## 实现说明

### 1. 算子工程生成

基于 `div_custom_template.json` 使用 `msopgen` 生成标准 Ascend C 自定义算子工程，生成 Host、Kernel、CMake 及自定义算子安装相关文件。

```bash
msopgen gen \
    -i div_custom_template.json \
    -c ai_core-ascend910b1 \
    -lan cpp \
    -out ./custom_op
```

### 2. Host 侧实现

在 `op_host/div_custom_template.cpp` 中完成算子注册、Shape 推导、数据类型推导和 Tiling 配置。

主要实现：

- 根据输入 Tensor Shape 计算总元素数量；
- 通过 TilingData 将总数据量 `size` 传递给 Kernel；
- 设置 `blockDim = 8`，由 8 个 AI Core 并行完成数据处理；
- `InferShape` 设置输出 Shape 与输入 `x` 保持一致；
- `InferDataType` 设置输出数据类型与输入保持一致；
- 支持 `float16` 和 `float32` 两种数据类型。

### 3. Kernel 侧实现

在 `op_kernel/div_custom_template.cpp` 中实现 Ascend C 核函数。

Kernel 采用：`CopyIn → Compute → CopyOut` 的处理流程。

其中：

- 使用 `GlobalTensor` 管理 Global Memory 中的输入输出数据；
- 使用 `LocalTensor` 和 `TQue` 管理 Unified Buffer 中的数据；
- 每个 Core 根据 `GetBlockIdx()` 计算自身负责的数据起始位置；
- 使用 `DataCopy` 将输入 `x`、`y` 从 GM 搬运到 UB；
- Compute 阶段调用 `AscendC::Div` 完成逐元素除法；
- 计算完成后将结果从 UB 搬回输出 Tensor `z`。

核心计算为：

```text
z[i] = x[i] / y[i]
```

通过 Double Buffer（BUFFER_NUM=2）实现搬运与计算的流水重叠。

### 目录结构

```
DivCustomTemplate/
├── div_custom_template.json       # 算子原型定义
├── run.sh                         # 一键编译运行脚本
├── test/
│   ├── main.cpp                   # ACLNN接口测试
│   └── CMakeLists.txt
└── custom_op/                     # msopgen生成的算子工程
    ├── CMakeLists.txt
    ├── CMakePresets.json
    ├── build.sh
    ├── framework/tf_plugin/
    ├── op_host/
    │   ├── CMakeLists.txt
    │   └── div_custom_template.cpp
    └── op_kernel/
        ├── CMakeLists.txt
        ├── div_custom_template_tiling.h
        └── div_custom_template.cpp
```

## 测试信息

实际运行结果符合预期：

```text
result is:
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5

test pass
```

验证结果：`test pass`

## 该PR关联的issue

fixes #

## 改动类型

- [x] 新功能 / New Feature

## 测试信息

- [x] 单元测试通过 / UT passed
- [x] 集成测试通过 / ST passed
- [x] 人工验证通过 / Manual verified

## 检查清单

- [x] 代码符合规范 / Code follows style guide
- [x] 测试添加并通过 / Tests added and passed
- [x] 文档已更新 / Docs updated if needed
- [x] 无硬编码敏感信息 / No secrets hardcoded
- [x] 提交信息符合规范 / Commit message follows convention
```

## 分支信息

- **源仓库**: `JeffDing/cann-outreach`
- **源分支**: `Challenge_04`
- **目标仓库**: `cann/cann-outreach`
- **目标分支**: `master`

## PR 创建链接

https://gitcode.com/cann/cann-outreach/pulls/new
