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

# Relu 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现 Relu 算子：对输入逐元素执行 ReLU。

| 项 | 规格 |
|---|---|
| 算子名 / 包名 | `Relu` |
| 输入 | 单输入 `x`，`DT_FLOAT` / `DT_FLOAT16`，`FORMAT_ND`，任意维 shape |
| 输出 | 单输出 `y`，与 `x` 同 shape、同 dtype |
| 属性 | 无 |
| 参考实现 | 逐元素 `y = max(x, 0)` |

**计算语义**：

```
y = max(x, 0)
```

**平台 golden**（`tests/ut/op_kernel/relu_data/gen_data.py`）：float64 中间计算后一次性转回输入 dtype。

**tilingKey**（`schMode`）：`0` → fp16/bf16 kernel，`1` → fp32 kernel（由 host 按输入 dtype 选择）。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/relu.h` | 实现 kernel 全部逻辑（模板类 `Relu<T>` 的 Init/Process/CopyIn/Compute/CopyOut） |
| `op_host/relu_tiling.cpp` | 实现 tiling 计算（平台信息查询、多核切分、UB 分块、tilingKey 选择） |
| `op_host/relu_infershape.cpp` | 输出 shape 与输入相同 |

**未修改任何其他文件**：`build.sh`、`CMakeLists.txt`、`op_host/relu_def.cpp`、`op_host/CMakeLists.txt`、`op_kernel/relu_tiling_data.h`（无 TODO）、`op_kernel/relu_tiling_key.h`、`op_kernel/relu.cpp`、`tests/`、`examples/` 全部保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`ReluTilingData` 保持原有 3 个 `int64_t` 字段，**标量直存、不做位域拆分**：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalNum` | N | 输入元素总数 |
| `blockFactor` | bf | 每核处理的元素数，按 32B 搬运粒度向上对齐 |
| `ubFactor` | uf | 每轮 UB 处理的元素数 |

host 侧计算（`op_host/relu_tiling.cpp:80-102`）：

1. `dataPerBlock = 32 / typeSize`（32 字节 DMA 粒度对应的元素数）；
2. `totalBlocks = CeilDiv(totalNum, dataPerBlock)`，`blocksPerCore = CeilDiv(totalBlocks, coreNum)`；
3. `blockFactor = blocksPerCore * dataPerBlock`，保证各核 GM 起始偏移满足 32B 对齐；
4. `ubFactor = FloorAlign(ubSize / (typeSize * bufferNum * 2), dataPerBlock)`，即双缓冲输入输出各占一半 UB；
5. `blockDim = CeilDiv(totalNum, blockFactor)`，消除空转核。

## 四、算法设计

### 4.1 infershape（`op_host/relu_infershape.cpp`）

- 输出 `y` 与输入 `x` 同 shape，直接 `*output_shape = *input_shape`；
- `input_shape == nullptr` 或 `output_shape == nullptr` 时返回 `ge::GRAPH_FAILED`。

### 4.2 host 侧 tiling（`op_host/relu_tiling.cpp`）

- **平台信息**：`GetPlatformInfo` 统一查询 `GetCoreNumAiv()` 与 `GetCoreMemSize(UB)`，为 0 时明确失败；
- **workspace = 0**：`WS_SYS_SIZE = 0U`，纯逐元素算子无跨核汇总需求；
- **dtype → typeSize**：fp16/bf16 取 2，其余取 4；
- **tilingKey**：fp16/bf16 → `RELU_TPL_SCH_MODE_0`，否则 → `RELU_TPL_SCH_MODE_1`。

### 4.3 kernel 侧（`op_kernel/relu.h`）

**每核处理区间**（`Init`）：`gmOffset = GetBlockIdx() * blockFactor`，两路 `SetGlobalBuffer` 绑定到该偏移。

**数据通路**：

- `TQue<VECIN, 2>` / `TQue<VECOUT, 2>` 双缓冲（`BUFFER_NUM = 2`）；
- buffer 长度 `bufLen = min(ubLength_, blockLength_)`，并做 `<= 0` 兜底；
- GM→UB / UB→GM 均为 `DataCopy`，`CopyIn(i, n)` / `CopyOut(i, n)` 的 GM 偏移为 `progress * ubLength_`。

**计算**（`Compute`）：`Maxs(yLocal, xLocal, static_cast<T>(0), currentNum)` 单指令完成逐元素 ReLU。

**尾块处理**（`Process`）：`tileNum = myCount / ubLength_` 个整轮 + `remainNum` 个尾块，尾块按剩余元素数截断。`myCount = totalNum - gmOffset` 再钳制到 `blockFactor`，`myCount <= 0` 直接返回（覆盖核数多于块数的空转核）。

## 五、关键决策依据

1. **blockFactor 按 32B 粒度对齐**：`blocksPerCore * dataPerBlock` 的形式保证多核场景各核 GM 起始偏移仍是 32B 整数倍，满足 MTE 搬运对齐要求，避免非对齐尾块走 `DataCopyPad` 慢路径。
2. **blockDim 由 blockFactor 反算**：`blockDim = CeilDiv(totalNum, blockFactor)`，小张量自动少开核，不存在空转核。
3. **bufLen 取 min(ubLength_, blockLength_)**：防止 `ubFactor > blockFactor` 时申请超出本核数据量的 UB（本地 UT 的占位 tiling 常见该组合）。
4. **workspace = 0**：逐元素算子无跨核归约，规避 workspace 汇总路径在 aclnn 调用下的指针不可靠风险。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 平台构建 `bash build.sh -j4` | **通过**（ascend910b） |
| CPU 仿真 UT（tikicpulib Ascend910B1）kernel UT | 通过 |
| CPU 仿真 UT sample verify | 通过 |

**UT 数据**（`tests/ut/op_kernel/relu_data/gen_data.py`）：输入 `np.ones((8, 2048))` 共 16384 元素，golden 为 float64 中间计算后转回输入 dtype；`compare_data.py` 逐元素比对。

## 七、性能要点

- 单指令 `Maxs` 完成逐元素计算，无标量回退；
- VECIN/VECOUT 双缓冲（深度 2），MTE2 搬运与 Vector 计算可流水重叠；
- `ubFactor` 取半池并按 32B 对齐，摊薄轮次调度开销；多核按 32B 块粒度切分，大张量充分并行。

## 八、约束遵守与交付状态

- 仅修改三个白名单文件（`op_kernel/relu.h`、`op_host/relu_tiling.cpp`、`op_host/relu_infershape.cpp`），未改动任何其他已有源码文件；
- `op_kernel/relu_tiling_data.h`、`op_kernel/relu_tiling_key.h`、`op_kernel/relu.cpp`、`op_host/relu_def.cpp` 均无 TODO，保持原样。
