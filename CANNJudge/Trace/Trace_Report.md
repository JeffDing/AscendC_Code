# Trace 算子实现报告

## 一、任务概述

在昇腾 NPU（Ascend C）上实现 Trace 算子：对 (M, N) 二维矩阵取主对角线 min(M, N) 个元素求和，输出 shape (1,)，dtype 与输入一致（fp16/fp32，内部 FP32 累加），M, N ∈ [1, 16384]，支持方阵/非方阵/退化场景（1×N、M×1、1×1）。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/trace.h` | 实现 kernel 全部逻辑（6 处 TODO，注释均保留） |
| `op_host/trace_tiling.cpp` | 实现 tiling 计算（2 处 TODO，注释保留） |
| `op_host/trace_infershape.cpp` | 输出 shape 固定为 (1,)（1 处 TODO，注释保留） |

**未修改任何其他文件**：`tests/` 目录全部文件、`op_kernel/trace_tiling_data.h`（无 TODO）、`op_kernel/trace.cpp`、`trace_tiling_key.h`、`trace_def.cpp` 均保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`TraceTilingData` 保持原有 3 个字段，语义约定如下：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalNum` | D = min(M, N) | 对角线元素总数 |
| `blockFactor` | bf | 每核处理的对角元数上限 |
| `ubFactor` | **N** | 矩阵列数（kernel 计算对角线 stride = N+1 的必需信息） |

- kernel 端每轮元素数 **F = min(bf, 1008)**（1008 为 UB 实测安全上限），核数充足时 bf ≤ 1008 每核单轮完成；核数不足时自动多轮，能力无损；
- kernel UT 的 1×1 用例赋值 `{1, 1, 1}` 在此语义下恰好解码正确（D=1 / bf=1 / N=1），**无需修改 test 文件**。

## 四、算法设计

### 4.1 kernel 侧（trace.h）

每核处理对角线区间 `[blockIdx * bf, +bf)`，按 F=1008 分轮，每轮：

1. **向量化 gather（CopyIn）**：`DataCopyPad(UB, gmX[(off+r)*(N+1)], {blockCount=cur, blockLen=sizeof(T), srcStride=N*sizeof(T), dstStride=0}, {pad=true})`——单条 DMA 硬件 gather 出 cur 个对角元（`blockLen < 32B` 时槽内为对角元同值广播，lane0 即对角元）。inQ 深度 2 实现 double-buffer 流水。
2. **掩码过滤 + FP32 归约（Compute）**：预构造 bit-mask（每 32B 槽仅 lane0 置位），`Mul` 过滤广播残留 → fp16 先 `Cast` fp32 → `ReduceSum` → 标量 `acc +=`（固定顺序，确定性）。
3. **输出（CopyOut）** 按 dtype 分档：
   - **fp32**：各核 `SetAtomicAdd<float>` 直接累加输出（累计误差 ~2^-24 ≪ 2^-13 阈值），真机 profiling 已验证（5 用例 5.8~7.7us 正常）；
   - **fp16**：host 固定单核调度，核内 FP32 精确累加，单次舍入 half 直写（零额外误差）；kernel 侧保留意外多核时的 half 原子加兜底分支。

block0 在 Init 清零输出后 `SyncAll()`，防御输出内存残留。

> **真机教训（v2 修改）**：初版 fp16 多核采用 workspace 汇总方案（各核 fp32 原子加到 `GetSysWorkSpacePtr()+RESERVED_WORKSPACE` 汇总槽），真机评测 profiling 实测该路径下 half kernel（blockDim=2）挂起，`aclrtSynchronizeStreamWithTimeout` 3 秒超时（507015）——API 返回了非空但无效的地址，判空防线无法拦截，CPU 仿真无法复现（仿真下返回 nullptr 走退化分支）。已删除全部 workspace 依赖代码，fp16 改为固定单核。

### 4.2 host 侧（trace_tiling.cpp）

- 解析 (M, N)：D = min(M, N)，`ubFactor = N`；
- 多核切分：float 走 `usedCore = min(coreNum, D)`、`bf = ceil(D/usedCore)` 并回修 usedCore 消除空转核；**fp16 固定 `usedCore = 1`**（见 4.1 真机教训）；
- workspace = 0（不使用 workspace）；
- tilingKey：fp16/bf16 → MODE_0(half kernel)，其余 → MODE_1(float kernel)。

### 4.3 infershape（trace_infershape.cpp）

输出 shape 固定 `SetDimNum(1); SetDim(0, 1)`。

## 五、关键决策依据（实验实证）

1. **fp16 输出策略（v2，真机实测驱动）**：初版 workspace 汇总方案真机挂起超时（见 4.1）；half 多核原子加经蒙特卡洛实证（RNE、2 万次×3 种子）64 核 27% 超 MERE 2^-10 → **fp16 固定单核**（核内 fp32 累加 + 1 次舍入，精度最优、零外部依赖）；float 多核 fp32 原子加真机已验证。
2. **bit-mask 陷阱**：mask 是 128 元素窗口（mask[0]→lane0-63，mask[1]→lane64-127），尾部槽 > 4 时第二个 64 位字才置位；mask 模式 `Duplicate` 不写未选中 lane 需先清零；repeat 间隔 256B。
3. **UB 容量**：池上限实测 196608B（192KB），fp32=128F+896、fp16=192F+640 → Fcap 统一取 1008（含 convBuf 单块 64KB、Duplicate repeatTime uint8 约束），边界值实测 PASS。
4. **N 的传递**：TilingData 结构不可改（无 TODO），3 字段装不下 4 个参数 → ubFactor 承载 N、F 由 kernel 端 min(bf, 1008) 推导，test 1×1 用例天然兼容。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| op_kernel UT + compare_data.py 精度比对 | **PASS**（MERE=0.000000, MARE=0.000000） |
| 正式实现探针（直接 include trace.h，fp16/fp32 全场景：1×1 / 1×16384 / 16384×1 / 33×99 / 37×100 / 100×37 / 16383×7 / 600×600 / 1000×1000 / 2048×2048 / 12000×9000 / 16384×16384 等） | **全部 PASS**（含 D=16384 单核 17 轮多轮场景） |
| op_host UT | **FAIL（预期内）**：恢复原样的期望值 `"0 1 0 "`、`{0}` 是空实现占位符（totalNum=0 意味着不处理任何元素，任何真实实现均无法匹配）；tilingKey=1 一项实际匹配 |

说明：CPU 仿真器（fork 模型）下并行多核原子加丢失更新、无参 SyncAll 不阻塞，为已知仿真器局限（真机为硬件实现）；多核正确性经"逐核串行 splitcheck + workspace 汇总链路探针"分别验证；官方 UT 单核场景不受影响。

## 七、性能要点

- 对角线 gather 单条 DMA 完成（无逐元素标量读取）；
- inQ 深度 2 double-buffer，MTE2 搬运与 Vector 计算流水重叠；
- F=1008 最大化摊薄轮次开销；多核均分对角线，核数 = min(coreNum, D) 无空转；
- UB 总占用 ≤ 192KB 池上限，只读对角元所在 32B 槽（不整矩阵搬运）。
