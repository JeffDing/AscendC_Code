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

# Tanh 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现 Tanh 算子：对输入逐元素执行双曲正切。

| 项 | 规格 |
|---|---|
| 算子名 / 包名 | `Tanh` |
| 输入 | 单输入 `x`，`DT_FLOAT` / `DT_FLOAT16`，`FORMAT_ND`，任意维 shape |
| 输出 | 单输出 `y`，与 `x` 同 shape、同 dtype |
| 属性 | 无 |
| 参考实现 | 逐元素 `tanh(x)` |

**计算语义**：

```
tanh(x) = 2 / (1 + exp(-2x)) - 1
```

kernel 未使用 `AscendC::Tanh` 内建指令，而是用 `Muls → Exp → Adds → Duplicate → Div → Adds` 六个基础向量指令组合实现（`op_kernel/tanh.h:96-103`），选用该形式的原因是它对 ±inf 输入数值稳定：

- `x → +inf` 时 `exp(-2x) → 0`，`2/1 - 1 = 1`；
- `x → -inf` 时 `exp(-2x) → inf`，`2/inf - 1 = -1`。

**tilingKey**（`schMode`）：`0` → fp16/bf16 kernel，`1` → fp32 kernel。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/tanh.h` | 实现 kernel 全部逻辑（模板类 `Tanh<T>` 的 Init/Process/CopyIn/Compute/CopyOut，含 tanh 指令组合） |
| `op_host/tanh_tiling.cpp` | 实现 tiling 计算（多核均分、UB 分块、tilingKey 选择） |
| `op_host/tanh_infershape.cpp` | 输出 shape 与输入相同 |

**未修改任何其他文件**：`build.sh`、`CMakeLists.txt`、`op_host/tanh_def.cpp`、`op_host/CMakeLists.txt`、`op_kernel/tanh_tiling_data.h`（无 TODO）、`op_kernel/tanh_tiling_key.h`、`op_kernel/tanh.cpp`、`tests/`、`examples/` 全部保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`TanhTilingData` 保持原有 3 个 `int64_t` 字段，标量直存：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalNum` | N | 输入元素总数 |
| `blockFactor` | bf | 每核处理的元素数 |
| `ubFactor` | uf | 每轮 UB 处理的元素数 |

host 侧计算（`op_host/tanh_tiling.cpp:89-104`）：

1. `blockFactor = CeilDiv(totalNum, coreNum)`，数据量过小（`< MIN_SPLIT_THRESHOLD = 1024` 且不足总量）时不拆核，`blockFactor = totalNum`；
2. `maxUbElems = (ubSize / UB_TOTAL_BUFFER_NUM) / typeSize`，其中 `UB_TOTAL_BUFFER_NUM = 4` 对应输入/输出各双缓冲共 4 份；
3. `maxUbElems` 按 `UB_ALIGN_BYTES = 32` 的 `FloorAlign` 取整；
4. `ubFactor = min(maxUbElems, blockFactor)`；
5. `blockDim = CeilDiv(totalNum, blockFactor)`。

## 四、算法设计

### 4.1 infershape（`op_host/tanh_infershape.cpp`）

输出 `y` 与输入 `x` 同 shape，直接 `*output_shape = *input_shape`，空指针保护。

### 4.2 host 侧 tiling（`op_host/tanh_tiling.cpp`）

- **平台信息**：`GetPlatformInfo` 查询 `GetCoreNumAiv()` 与 `GetCoreMemSize(UB)`；
- **typeSize**：`ge::GetSizeByDataType`，`<= 0` 明确失败；
- **workspace = 0**：`WS_SYS_SIZE = 0U`；
- **tilingKey**：fp16/bf16 → `TANH_TPL_SCH_MODE_0`，否则 → `TANH_TPL_SCH_MODE_1`。

### 4.3 kernel 侧（`op_kernel/tanh.h`）

**每核处理区间**（`Init`）：`startOffset = blockFactor * GetBlockIdx()`，`blockLength_ = min(totalNum - startOffset, blockFactor)`，负数 remainder 收敛到 0。

**UB 分块**（`Init`）：`maxUbElems = UB_CHUNK_BYTES / sizeof(T)`，其中 `UB_CHUNK_BYTES = 32 * 1024` 为**单次循环安全字节数上限**；`ubLength_` 越界或超限时一律回退 `maxUbElems`。32KB × 4 份 = 128KB，低于 192KB UB 池，无论平台 UB 为 192KB 还是仿真器环境都不会溢出。

**数据通路**：

- `TQue<VECIN, 2>` / `TQue<VECOUT, 2>` 双缓冲；
- GM→UB / UB→GM 均为 `DataCopy`，偏移 `progress * ubLength_`。

**计算**（`Compute`，`:96-106`）：

```cpp
Muls(yLocal, xLocal, -2.0, currentNum);   // -2x
Exp(yLocal, yLocal, currentNum);          // exp(-2x)
Adds(yLocal, yLocal, 1.0, currentNum);    // 1 + exp(-2x)
Duplicate(xLocal, 2.0, currentNum);       // 复用输入缓冲存常量 2，避免额外占 UB
Div(xLocal, xLocal, yLocal, currentNum);  // 2 / (1 + exp(-2x))
Adds(xLocal, xLocal, -1.0, currentNum);   // - 1
DataCopy(yLocal, xLocal, currentNum);     // 结果搬回输出队列
```

**循环**（`Process`）：`for (i = 0; i < blockLength_; i += ubLength_)`，尾块 `currentNum = blockLength_ - i` 截断。

## 五、关键决策依据

1. **不用 `AscendC::Tanh` 内建，改用 6 指令组合**：内建 `Tanh` 的中间精度路径与 golden（float64 中间后单次舍入）存在系统性偏差风险；`Muls/Exp/Adds/Div` 组合的中间过程全在目标精度域内，配合单次输出舍入可与 golden 对齐。同时该形式对 ±inf 天然稳定，无需 Compare/Select 分支。
2. **`UB_CHUNK_BYTES = 32KB` 常量兜底而非按 ubSize 计算**：host 传的 `ubFactor` 可能因占位 tiling 远超 UB 容量，kernel 侧用固定 32KB 常量钳制是最简防御；32KB × 4 份 = 128KB 在 192KB 与 256KB 池上都有裕量。
3. **复用 `xLocal` 存常量 2**：`Div` 前输入 `xLocal` 数据已用完，用 `Duplicate` 覆写为常量 2，省去第三份 UB 缓冲，使总占用仍是 4 份而非 5 份。
4. **小张量不拆核**（`MIN_SPLIT_THRESHOLD = 1024`）：`blockFactor < 1024 && blockFactor < totalNum` 时合并为整块，避免 (1,1) 类小张量起多核的空转开销。
5. **workspace = 0**：逐元素算子无跨核归约。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 平台构建 `bash build.sh -j4` | **通过**（ascend910b） |
| CPU 仿真 UT（tikicpulib Ascend910B1）kernel UT | 通过 |
| CPU 仿真 UT sample verify | 通过 |

**UT 数据**（`tests/ut/op_kernel/tanh_data/gen_data.py`）：输入 `np.ones((8, 2048))` 共 16384 元素，golden 为 float64 中间计算后转回输入 dtype；`compare_data.py` 逐元素比对。

## 七、性能要点

- 6 条向量指令完成单元素 tanh，比 `Exp`+多指令的其他实现更紧凑，但比单条内建 `Tanh` 多约 5 条指令；
- VECIN/VECOUT 双缓冲（深度 2），MTE2 搬运与 Vector 计算流水重叠；
- `ubLength_` 上限 32KB/元素类型（fp32 8192 元素 / fp16 16384 元素），单轮覆盖平台用例全部数据；
- 多核按 `CeilDiv(totalNum, coreNum)` 均分，小张量退化为单核。

## 八、约束遵守与交付状态

- 仅修改三个白名单文件（`op_kernel/tanh.h`、`op_host/tanh_tiling.cpp`、`op_host/tanh_infershape.cpp`），未改动任何其他已有源码文件；
- `op_kernel/tanh_tiling_data.h`、`op_kernel/tanh_tiling_key.h`、`op_kernel/tanh.cpp`、`op_host/tanh_def.cpp` 均无 TODO，保持原样。
