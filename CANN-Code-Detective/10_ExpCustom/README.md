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

# Exp 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现 Exp 算子：对输入逐元素计算自然指数。

| 项 | 规格 |
|---|---|
| 算子名 / 包名 | `Exp` |
| 输入 | 单输入 `x`，`DT_FLOAT` / `DT_FLOAT16`，`FORMAT_ND`，任意维 shape |
| 输出 | 单输出 `y`，与 `x` 同 shape、同 dtype |
| 属性 | 无 |
| 参考实现 | 逐元素 `exp(x)` |

**计算语义**：

```
y = exp(x)
```

**平台 golden**（`tests/ut/op_kernel/exp_data/gen_data.py`）：`np.exp(x.astype(np.float64))` 后一次性 `astype` 回输入 dtype，即 **float64 中间精度**。这决定了 kernel 应采用较高精度的 exp 路径。

**tilingKey**（`schMode`）：`0` → fp16/bf16 kernel，`1` → fp32 kernel。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/exp.h` | 实现 kernel 全部逻辑（模板类 `Exp<T>` 的 Init/Process/CopyIn/Compute/CopyOut） |
| `op_host/exp_tiling.cpp` | 实现 tiling 计算（UB 3/4 预算、32B 对齐多核切分、tilingKey 选择） |
| `op_host/exp_infershape.cpp` | 输出 shape 与输入相同 |

**未修改任何其他文件**：`build.sh`、`CMakeLists.txt`、`op_host/exp_def.cpp`、`op_host/CMakeLists.txt`、`op_kernel/exp_tiling_data.h`（无 TODO）、`op_kernel/exp_tiling_key.h`、`op_kernel/exp.cpp`、`tests/`、`examples/` 全部保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`ExpTilingData` 保持原有 3 个 `int64_t` 字段，标量直存：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalNum` | N | 输入元素总数 |
| `blockFactor` | bf | 每核处理的元素数，按 32B 粒度向上对齐 |
| `ubFactor` | uf | 每轮 UB 处理的元素数 |

host 侧计算（`op_host/exp_tiling.cpp:97-121`）：

1. `typeSize` = 2（fp16/bf16）或 4（fp32）；
2. `alignNum = GetUbBlockSize(context) / typeSize`，即 32B DMA 粒度对应的元素数；
3. **核间切分**：`totalNum < MIN_SPLIT_THRESHOLD(1024)` 时 `blockDim = 1` 不拆核；否则 `blockDim = min(coreNum, totalNum)`；
   `blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum)`，再以 `blockFactor` 反算 `blockDim = max(1, CeilDiv(totalNum, blockFactor))`；
4. **核内切分**：`usableUbSize = ubSize * 3 / 4`（预留 1/4 系统开销，与 CPU 仿真池上限 0.75×UB_SIZE 一致），
   `ubTotalNum = FloorDiv(usableUbSize, typeSize * QUEUE_NUM(2) * BUFFER_NUM_IN_UB(2))`，
   `ubFactorCap = max(alignNum, FloorAlign(ubTotalNum, alignNum))`，最终 `ubFactor = min(blockFactor, ubFactorCap)`。

## 四、算法设计

### 4.1 infershape（`op_host/exp_infershape.cpp`）

输出 `y` 与输入 `x` 同 shape，`*output_shape = *input_shape`，空指针返回 `GRAPH_FAILED`。

### 4.2 kernel 侧（`op_kernel/exp.h`）

**每核处理区间**（`Init`，`:57-90`）：`blockOffset = GetBlockIdx() * blockFactor`，
`blockLength_ = clamp(totalNum - blockOffset, 0, blockFactor)`，尾核数据不足时自动截断。

**UB 收敛三重保护**（`Init`，`:66-83`）：

```cpp
const int64_t ubLimit = UB_TOTAL_BYTES * UB_USABLE_NUM / UB_USABLE_DEN /
    (sizeof(T) * UB_QUEUE_NUM * BUFFER_NUM);   // 192KB 预算
int64_t ubFactor = tilingData->ubFactor;
if (ubFactor > ubLimit) ubFactor = ubLimit;    // ① 钳制到 UB 预算
ubFactor = ubFactor / alignElems * alignElems; // ② 按 32B 向下对齐
if (ubFactor < alignElems) ubFactor = alignElems;  // ③ 至少一个对齐块
ubLength_ = min(blockLength_, ubFactor);
```

其中 `UB_TOTAL_BYTES = 262144`（910B UB 共 256KB）、`UB_USABLE_NUM/DEN = 3/4`。
`blockLength_ <= 0 || ubLength_ <= 0` 时两字段置 0 并提前返回，`Process` 首行同样守卫，保证退化 tiling 下不越界、不死循环。

**数据通路**（含非对齐尾块分支，`CopyIn`/`CopyOut`）：

- 字节数按 32B 对齐 → `DataCopy`（连续 burst，最快路径）；
- 非对齐 → `DataCopyPad` + `DataCopyExtParams{1, currentNum*sizeof(T), 0, 0, 0}`，入方向补 `DataCopyPadExtParams<T>()`，出方向用 3 参重载。

**计算**（`Compute`）：`AscendC::Exp(yLocal, xLocal, expCount)` 内建指令。

**循环**（`Process`）：`while (remainNum > 0)` 逐轮 `CopyIn → Compute → CopyOut`，`currentNum = min(remainNum, ubLength_)`。

## 五、关键决策依据

1. **UB 预算 3/4 而非全池**：CPU 仿真池上限即 `0.75 * UB_SIZE`，host 与 kernel 两侧用同一比例可保证仿真与真机行为一致，同时给框架留 1/4 余量。
2. **blockFactor 必须按 32B 向上对齐**：多核场景下各核 GM 起始偏移 = `blockIdx * blockFactor`，若不对齐则后续核的起始地址非 32B 整数倍，MTE 搬运走慢路径甚至失败。先 `CeilAlign` 再反算 `blockDim` 是标准做法（反算后核数可能略少于 `coreNum`，但每核边界一定对齐）。
3. **kernel 侧二次钳制 ubFactor**：host 的 `ubFactor` 已按预算计算，但本地 UT 使用占位 tiling（如 `ubFactor = 16384`）可能远超 UB 容量；kernel 侧独立钳制使任何 tiling 输入下都不越界，兼顾"算法正确"与"UT 不崩"。
4. **尾块按字节数判定对齐**：`(currentNum * sizeof(T)) % 32 == 0` 是元素数判定的等价形式，直接按字节比较避免 `sizeof(T)` 整除歧义；非对齐走 `DataCopyPad` 而非截断，保证尾部元素不丢。
5. **workspace = 0**：逐元素算子无跨核汇总需求。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 平台构建 `bash build.sh -j4` | **通过**（ascend910b） |
| CPU 仿真 UT（tikicpulib Ascend910B1）kernel UT | 通过 |
| CPU 仿真 UT sample verify | 通过 |

**UT 数据**（`tests/ut/op_kernel/exp_data/gen_data.py`）：输入 `np.ones((8, 2048))` 共 16384 元素，golden = `exp(1.0) ≈ 2.7182817`；`compare_data.py` 逐元素比对。UT 的 `tilingData` 为 `{totalNum=16384, blockFactor=16384, ubFactor=16384}`、`ICPU_SET_TILING_KEY(1)`（fp32 路径）、`numBlocks=1`。

## 七、性能要点

- 单指令 `AscendC::Exp` 完成逐元素计算；
- VECIN/VECOUT 双缓冲（深度 2），MTE2 搬运与 Vector 计算流水重叠；
- `ubFactor` 取 3/4 池（fp32 约 24576 元素 / fp16 约 49152 元素），单轮覆盖 16384 元素的平台用例，循环开销趋零；
- 多核按 32B 块粒度对齐切分，大张量充分并行；`< 1024` 元素不拆核避免空转。

## 八、约束遵守与交付状态

- 仅修改三个白名单文件（`op_kernel/exp.h`、`op_host/exp_tiling.cpp`、`op_host/exp_infershape.cpp`），未改动任何其他已有源码文件；
- `op_kernel/exp_tiling_data.h`、`op_kernel/exp_tiling_key.h`、`op_kernel/exp.cpp`、`op_host/exp_def.cpp` 均无 TODO，保持原样。
