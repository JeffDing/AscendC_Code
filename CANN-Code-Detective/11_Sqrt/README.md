# 算子工程调测

加载 CANN 9.0.0 或 9.1.0 环境，安装 Python 参考实现需要的 numpy、ml_dtypes（以及参考文件实际使用的 torch/tensorflow）。

```bash
source /path/to/cann/set_env.sh
bash build.sh -j4       # 构建 RUN 包
DEVICE_ID=0 bash build.sh -j4 -e  # 构建并运行首个测试点、比较输出
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include GTEST_SOURCE_DIR=/path/to/googletest bash build.sh -j4 -u # 本地 UT
```

GoogleTest 使用本地源码（建议1.14.0），nlohmann/json 头文件建议3.11.3（include目录下应有nlohmann/json.hpp），不自动下载。CPU 调测还需要工具链的 tikicpulib。修改 op_host、op_kernel，以及存在时的 op_api。kernel为空时输出校验失败是正常现象。samples 只包含第一个 NPU 测试点。输入按题目规范生成，比较前检查文件大小。

opapi 工程的完整结果由外层 NPU example 校验；内层 CPU UT 的 fixture 需要按实际内外层转换关系自行提供，不能直接套外层 GT。默认 -u 明确跳过内层数值断言。准备并编辑内层 fixture 后可使用 `ENABLE_INNER_KERNEL_UT=ON bash build.sh -u` 启用；其结果需按内层语义自行断言。API UT 需要设备上下文，默认明确跳过；先构建 RUN，再使用 `RUN_OP_API_UT=1 DEVICE_ID=0 bash build.sh -u` 执行外层 workspace 接口测试，该测试不代替 -e 的数值校验。若修改 tiling 结构，也需要同步调整 kernel UT 的 tiling fixture。

---

# Sqrt 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现 Sqrt 算子：对输入逐元素计算平方根。

| 项 | 规格 |
|---|---|
| 算子名 / 包名 | `Sqrt` |
| 输入 | 单输入 `x`，`DT_FLOAT` / `DT_FLOAT16`，`FORMAT_ND`，任意维 shape |
| 输出 | 单输出 `y`，与 `x` 同 shape、同 dtype |
| 属性 | 无 |
| 参考实现 | 逐元素 `sqrt(x)` |

**计算语义**：

```
y = sqrt(x)
```

**平台 golden**（`tests/ut/op_kernel/sqrt_data/gen_data.py`）：float64 中间计算后一次性转回输入 dtype。

**tilingKey**（`schMode`）：`0` → fp16/bf16 kernel，`1` → fp32 kernel。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/sqrt.h` | 实现 kernel 全部逻辑（模板类 `Sqrt<T>`，含头文件自包含冲突的 guard 预处理） |
| `op_host/sqrt_tiling.cpp` | 实现 tiling 计算（UB 3/4 预算、32B 对齐多核切分、tilingKey 选择） |
| `op_host/sqrt_infershape.cpp` | 输出 shape 与输入相同 |

**未修改任何其他文件**：`build.sh`、`CMakeLists.txt`、`op_host/sqrt_def.cpp`、`op_host/CMakeLists.txt`、`op_kernel/sqrt_tiling_data.h`（无 TODO）、`op_kernel/sqrt_tiling_key.h`、`op_kernel/sqrt.cpp`、`tests/`、`examples/` 全部保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`SqrtTilingData` 保持原有 3 个 `int64_t` 字段，标量直存：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalNum` | N | 输入元素总数 |
| `blockFactor` | bf | 每核处理的元素数，按 32B 粒度向上对齐 |
| `ubFactor` | uf | 每轮 UB 处理的元素数 |

host 侧计算与 Exp 完全同构（`op_host/sqrt_tiling.cpp:98-118`）：`alignNum = GetUbBlockSize/typeSize`；`totalNum < 1024` 单核否则 `min(coreNum, totalNum)`；`blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum)` 后反算 `blockDim`；`ubFactor = min(blockFactor, max(alignNum, FloorAlign(FloorDiv(ubSize*3/4, typeSize*2*2), alignNum)))`。

## 四、算法设计

### 4.1 infershape（`op_host/sqrt_infershape.cpp`）

输出 `y` 与输入 `x` 同 shape，`*output_shape = *input_shape`，空指针返回 `GRAPH_FAILED`。

### 4.2 kernel 侧（`op_kernel/sqrt.h`）

**头文件自包含冲突的规避**（`:6-13`）：CANN 9.0.0 内部头 `<asc/impl/utils/std/cmath/sqrt.h>` 在部分工具链的包含链里会 `#include "sqrt.h"`，而 opc 预编译把源码目录置于 `-I` 前端，导致递归包含自身并报 `#include nested too deeply`。解法是提前置灰该内部头的 guard：

```cpp
#ifndef IMPL_STD_SQRT_H
#define IMPL_STD_SQRT_H
#endif
```

使其退化为空壳。本 kernel 使用向量级 `AscendC::Sqrt(yLocal, xLocal, count)`，不依赖 `AscendC::Std::sqrt`，压制该头不影响功能。

**每核处理区间与 UB 收敛**（`Init`，`:67-101`）：与 Exp 同构——`blockOffset = GetBlockIdx() * blockFactor`，`blockLength_` 截断到 `[0, blockFactor]`；`ubFactor` 依次做 UB 预算钳制（192KB = 256KB×3/4）、32B 向下对齐、下限保护，再与 `blockLength_` 取 min；`blockLength_ <= 0 || ubLength_ <= 0` 置 0 返回。

**数据通路**（`CopyIn`/`CopyOut`）：字节数 32B 对齐走 `DataCopy`，非对齐走 `DataCopyPad` + `DataCopyExtParams{1, currentNum*sizeof(T), 0, 0, 0}`（入方向补 `DataCopyPadExtParams<T>()`，出方向 3 参重载）。

**计算**（`Compute`）：`AscendC::Sqrt(yLocal, xLocal, sqrtCount)` 内建指令。

**循环**（`Process`）：`while (remainNum > 0)`，`currentNum = min(remainNum, ubLength_)`。

## 五、关键决策依据

1. **guard 预处理优先于改 include**：自包含冲突的根因是工具链 `-I` 顺序把源码目录排在内部头之前，无法通过调整本工程的 include 语句解决；置灰内部头 guard 是唯一不动构建配置的办法，且因不使用 `Std::sqrt` 而无副作用。
2. **UB 3/4 预算与 32B 对齐**：与 Exp 同一套设计，保证 host 计算值与 kernel 实际消耗一致，且 CPU 仿真池上限（0.75×UB_SIZE）不会误判溢出。
3. **kernel 侧二次钳制**：本地 UT 的占位 `ubFactor` 可能远超 UB 容量，kernel 独立钳制后任何 tiling 输入下都不越界。
4. **blockFactor 32B 向上对齐 + 反算 blockDim**：保证多核 GM 起始偏移对齐。
5. **workspace = 0**：逐元素算子无跨核汇总。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 平台构建 `bash build.sh -j4` | **通过**（ascend910b） |
| CPU 仿真 UT（tikicpulib Ascend910B1）kernel UT | 通过 |
| CPU 仿真 UT sample verify | 通过 |

**UT 数据**（`tests/ut/op_kernel/sqrt_data/gen_data.py` + `compare_data.py`）：输入 `np.ones((8, 2048))` 共 16384 元素，golden 为 float64 中间计算后转回输入 dtype。

## 七、性能要点

- 单指令 `AscendC::Sqrt` 完成逐元素计算；
- VECIN/VECOUT 双缓冲（深度 2），MTE2 搬运与 Vector 计算流水重叠；
- `ubFactor` 取 3/4 池，单轮覆盖平台用例全部数据；
- 多核按 32B 块粒度对齐切分，`< 1024` 元素不拆核。

## 八、约束遵守与交付状态

- 仅修改三个白名单文件（`op_kernel/sqrt.h`、`op_host/sqrt_tiling.cpp`、`op_host/sqrt_infershape.cpp`），未改动任何其他已有源码文件；
- `op_kernel/sqrt_tiling_data.h`、`op_kernel/sqrt_tiling_key.h`、`op_kernel/sqrt.cpp`、`op_host/sqrt_def.cpp` 均无 TODO，保持原样；
- kernel 头部的 `IMPL_STD_SQRT_H` guard 块是为绕过工具链包含冲突而增加的预处理，不含 TODO，随 `sqrt.h` 一并交付。
