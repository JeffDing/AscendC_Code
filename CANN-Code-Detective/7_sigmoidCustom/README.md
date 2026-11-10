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

# Sigmoid 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现 Sigmoid 算子：对输入逐元素执行 sigmoid。

| 项 | 规格 |
|---|---|
| 算子名 / 包名 | `Sigmoid` |
| 输入 | 单输入 `x`，`DT_FLOAT` / `DT_FLOAT16`，`FORMAT_ND`，任意维 shape |
| 输出 | 单输出 `y`，与 `x` 同 shape、同 dtype |
| 属性 | 无 |
| 参考实现 | 逐元素 `1 / (1 + exp(-x))` |

**计算语义**：

```
y = 1 / (1 + exp(-x))
```

**平台 golden**（`tests/ut/op_kernel/sigmoid_data/gen_data.py`）：float64 中间计算 `1.0 / (1.0 + np.exp(-x.astype(np.float64)))` 后一次性转回输入 dtype。

**tilingKey**（`schMode`）：`0` → fp16/bf16 kernel，`1` → fp32 kernel（由 host 按输入 dtype 选择）。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/sigmoid.h` | 实现 kernel 全部逻辑（模板类 `Sigmoid<T>` 的 Init/Process/CopyIn/Compute/CopyOut） |
| `op_host/sigmoid_tiling.cpp` | 实现 tiling 计算（多核切分、UB 分块、tilingKey 选择） |
| `op_host/sigmoid_infershape.cpp` | 输出 shape 与输入相同 |

**未修改任何其他文件**：`build.sh`、`CMakeLists.txt`、`op_host/sigmoid_def.cpp`、`op_host/CMakeLists.txt`、`op_kernel/sigmoid_tiling_data.h`（无 TODO）、`op_kernel/sigmoid_tiling_key.h`、`op_kernel/sigmoid.cpp`、`tests/`、`examples/` 全部保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`SigmoidTilingData` 保持原有 3 个 `int64_t` 字段，**标量直存、不做位域拆分**：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalNum` | N | 输入元素总数 |
| `blockFactor` | bf | 每核处理的元素数 |
| `ubFactor` | uf | 每轮 UB 处理的元素数 |

host 侧计算（`op_host/sigmoid_tiling.cpp:71-106`）：

1. `totalNum` = 输入 shape 各维相乘；
2. `block_dim = coreNum`，若 `totalNum < coreNum` 则退化为 1；`blockFactor = CeilDiv(totalNum, block_dim)`；
3. `available_ub = ubSize - 16KB`（预留系统开销，负值兜底回 `ubSize`）；
4. `max_elements = available_ub / (type_size * 3)`，`ubFactor = FloorAlign(max_elements, 32)` 并按 `blockFactor` 收敛；
5. `ubFactor < 32` 时兜底为 32。

**`type_size * 3` 的含义**：输入队列 + 输出队列 + AscendC `Sigmoid` 内建指令所需的中间工作区，共 3 份缓冲。

## 四、算法设计

### 4.1 infershape（`op_host/sigmoid_infershape.cpp`）

输出 `y` 与输入 `x` 同 shape，直接 `*output_shape = *input_shape`，空指针保护返回 `GRAPH_FAILED`。

### 4.2 host 侧 tiling（`op_host/sigmoid_tiling.cpp`）

- **平台信息**：`GetPlatformInfo` 查询 `GetCoreNumAiv()` 与 `GetCoreMemSize(UB)`；
- **workspace = 0**：`WS_SYS_SIZE = 0U`；
- **type_size**：fp16 取 2，其余取 4；
- **tilingKey**：fp16/bf16 → `SIGMOID_TPL_SCH_MODE_0`，否则 → `SIGMOID_TPL_SCH_MODE_1`。

### 4.3 kernel 侧（`op_kernel/sigmoid.h`）

**每核处理区间**（`Init`）：`remainder = totalNum - blockFactor * blockIdx`，`blockLength_ = min(remainder, blockFactor)`（负数 remainder 天然截断为 0 或负值），两路 `SetGlobalBuffer` 偏移 `blockFactor * blockIdx`。

**数据通路**：

- `TQue<VECIN, 1>` / `TQue<VECOUT, 1>`，`BUFFER_NUM = 1`（单缓冲）；
- buffer 长度 `ubLength_ * sizeof(T)`；
- GM→UB：`DataCopyPad` + `DataCopyExtParams{1, currentNum*sizeof(T), 0, 0, 0}` + `DataCopyPadExtParams<T>{pad=false}`；
- UB→GM：`DataCopyPad` + `DataCopyExtParams` 3 参形式。

**计算**（`Compute`）：`AscendC::Sigmoid(localY, localX, currentNum)` 内建指令单步完成。

**循环**（`Process`）：`totalProgress = CeilDiv(blockLength_, ubLength_)`，末轮 `currentNum = blockLength_ - progress * ubLength_` 处理尾块。

## 五、关键决策依据

1. **UB 预算按 `type_size * 3` 而非 `* 2`**：AscendC `Sigmoid` 内建指令内部需要额外一份中间工作区，若只按输入/输出两份预算会挤占该工作区导致 UB 溢出。这是与 Relu（`Maxs` 无中间区）的关键差异。
2. **BUFFER_NUM = 1 单缓冲**：sigmoid 的 UB 需求已是 3 份/元素，若再用双缓冲会显著压缩每轮处理元素数；单缓冲下每轮 `ubFactor` 可达 `(ubSize-16KB)/12`，fp32 下约 21K 元素，单轮即可覆盖平台用例全部 16384 元素。
3. **available_ub 预留 16KB**：`ubSize - 16 * 1024` 给框架/编译器预留头部空间，负值时回退全池避免小 UB 平台算出负数。
4. **ubFactor 按 32 对齐并受 blockFactor 收敛**：保证 GM 偏移 32B 对齐，同时避免单轮处理量超过本核数据量造成无效循环。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 平台构建 `bash build.sh -j4` | **通过**（ascend910b） |
| CPU 仿真 UT（tikicpulib Ascend910B1）kernel UT | 通过 |
| CPU 仿真 UT sample verify | 通过 |

**UT 数据**（`tests/ut/op_kernel/sigmoid_data/gen_data.py`）：输入 `np.ones((8, 2048))` 共 16384 元素，golden 为 `1/(1+exp(-1)) ≈ 0.7310586`。

**调测残留文件说明**：`npuchk/`、`cceprint/`、`stub_reg.log` 为本地仿真调测过程产生的日志与 stub 注册记录（记录 AscendC 内建函数在仿真器中的符号解析结果，多数 `copy_*`/`matrix_*` 接口为 `(nil)`，属仿真环境正常现象），不影响算子功能，可酌情清理。

## 七、性能要点

- `AscendC::Sigmoid` 内建指令单步完成，无 `Exp` + `Adds` + `Div` 三段手写组合；
- 单缓冲 + 大 `ubFactor`（fp32 约 21K 元素），平台用例单轮处理完毕，循环开销趋近于零；
- 多核按 `CeilDiv(totalNum, coreNum)` 均分，`totalNum < coreNum` 时退化为单核避免空转。

## 八、约束遵守与交付状态

- 仅修改三个白名单文件（`op_kernel/sigmoid.h`、`op_host/sigmoid_tiling.cpp`、`op_host/sigmoid_infershape.cpp`），未改动任何其他已有源码文件；
- `op_kernel/sigmoid_tiling_data.h`、`op_kernel/sigmoid_tiling_key.h`、`op_kernel/sigmoid.cpp`、`op_host/sigmoid_def.cpp` 均无 TODO，保持原样。
