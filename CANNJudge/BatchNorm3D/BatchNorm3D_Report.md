# BatchNorm3D 算子实现报告

## 一、任务概述

在昇腾 NPU（Ascend C）上实现 **BatchNorm3D** 算子（训练模式，单算子同时完成统计与归一化）：

- 输入：`x` (N, C, D, H, W)（float16 / float32）、`scale` (C,) / `offset` (C,)（均为 float32，即 γ / β）；属性 `epsilon`（默认 1e-5）。
- 输出：`y` 与 `x` 同 shape / 同 dtype；`save_mean`、`save_variance` 为 (C,) float32。
- 计算（与题目 `gen_data.py` 的 numpy 参考实现一致，沿 (N, D, H, W) 做有偏方差）：
  `N_c = N*D*H*W`，`μ_c = (1/N_c)·Σx`，`σ_c² = (1/N_c)·Σx² − μ_c²`，
  `y_c = (x_c − μ_c)/sqrt(σ_c²+ε) · γ_c + β_c`。
- 支持方阵/退化场景：任意 N/C/D/H/W ≥ 1，含 D·H·W=1、C=1、跨 UB 多块等。

## 二、改动清单（仅限 3 个含 TODO 的文件）

| 文件 | 改动内容 |
|---|---|
| `op_host/batch_norm3_d_infershape.cpp` | 形状推导：`y=x`，`save_mean/save_variance=(C,)`（1 处 TODO，注释保留） |
| `op_host/batch_norm3_d_tiling.cpp` | Tiling：解析 5D、多核按通道切分、按位打包 TilingData、tilingKey（2 处 TODO，注释保留） |
| `op_kernel/batch_norm3_d.h` | kernel 全部逻辑（6 处 TODO，注释均保留） |

**未修改任何其他文件**：`op_kernel/batch_norm3_d.cpp`、`*_tiling_data.h`、`*_tiling_key.h`、`op_host/*_def.cpp`、`tests/`、`examples/`、CMake 配置均保持原样（测试运行生成的中间 `.bin` 已清理）。

## 三、TilingData 三字段的语义约定（结构未动）

`BatchNorm3DTilingData` 保持原有 3 个 `int64_t` 字段，host / kernel 两侧严格一致：

| 字段 | 位域 | 语义 |
|---|---|---|
| `totalNum` | — | 保留 = N·C·DHW（kernel 不读取该字段） |
| `blockFactor` | `[0:39)` = DHW = D·H·W，`[39:52)` = C | 关键参数打包 |
| `ubFactor` | `[0:32)` = epsilon 的 IEEE-754 位模式，`[32:64)` = N | 属性/参数下发 |

- kernel 由 `DHW/C/N` 自行推导元素总数与每通道归约集合大小，**不读取 `totalNum`**（沿用 BNTrainingUpdate 的真机教训：第一字段在 kernel 侧读取不可靠）；
- 单块 UB 元素数用编译期常量 `BN3D_UB_ELEMS = 4096`，恒定不超过 UB 容量，无需 host 下发。

## 四、算法设计

### 4.1 kernel 侧（batch_norm3_d.h）

输入按 (N,C,D,H,W) 行主序存储，对固定通道 `c`，元素被分为 `N` 段、每段 `DHW` 个连续元素。**按通道将工作切分到各核**（通道间统计彼此独立，无跨核依赖）：

1. **pass 1（统计）**：本核遍历其每个通道的全部 N 段，逐段按 `BN3D_UB_ELEMS` 分块 `CopyIn`（`TQue<VECIN>` 双缓冲），fp16 先 `Cast` 升 fp32，做向量累加 `acc += x`、`accSq += x·x`；累加器**跨段不复位**——段内以 `o` 为块起点，块内位置 `p` 映射到 `acc[p % 4096]`，各段起始均回到 `o=0`，故每个元素恰好被累加一次，全部段处理完仅需一次 `ReduceSum` 即得整通道 `sum/sumsq`。随即 `mean=sum/N_c`、`var=sumsq/N_c−mean²`（夹紧非负），`invStd=1/sqrt(var+ε)`（设备端 `sqrt`）。
2. **pass 2（归一化 + 仿射）**：`coefA=invStd·γ`、`coefB=β−mean·coefA`，逐段分块 `CopyIn → Muls/Adds（fp16 先升 fp32 再降回）→ CopyOut`（`TQue<VECOUT>`）。
3. **统计量写出**：本核各通道 `mean/var` 先 `SetValue` 写入 UB 缓冲，循环结束后统一经 **MTE3 `DataCopyPad`** 写出到 `save_mean/save_variance`（`S_MTE3` 同步），末尾 `MTE3_MTE2` 等待保证写出完成。

### 4.2 host 侧（batch_norm3_d_tiling.cpp）

- 解析 x 的 (N,C,D,H,W)（兼容 1~4 维退化）；用 `scale` 的 (C,) 校正通道数；
- `dhw = D·H·W`、`totalNum = N·C·dhw`；读取 `epsilon`；按输入 dtype 设 tilingKey（fp16/bf16→MODE_0，fp32→MODE_1）；
- 多核：`blockDim = min(coreNum, C)`，按通道均分，无空转核；workspace = 0。

### 4.3 infershape（batch_norm3_d_infershape.cpp）

`y ← x`；`save_mean/save_variance ← scale` 的 (C,)（退化时由 x 第 1 维推导）。

## 五、关键决策依据（真机实验实证）

> 以下为在本机 Ascend 910B3（CANN 9.0.0）上用 aclnn 两段接口实测得出的**非显性结论**。

1. **多核 `GlobalTensor::SetValue` 直写 GM 会大面积丢失（本次最大陷阱）**：初版在每个核内直接用 `saveMeanGM.SetValue(c, mean)` 写统计量。真机上 **`y` 完全正确**（内部标量 mean/var 正确），但 `save_mean` 几乎全为 0——聚焦实验（C=16、blockDim=16、每核 1 通道、输入 x=1..16）读回 `save_mean=[0,0,3,0,0,6,0,0,…]`，仅通道 2/5 落盘。即**多核标量直写 GM 未可靠刷新**。修复：本核各通道统计量先写入 UB，循环结束后统一用 **MTE3 `DataCopyPad`** 写出（`S_MTE3` 同步 + 末尾 `MTE3_MTE2` 等待）。改后聚焦实验 `save_mean=[1..8]` 正确，批量 34/34 全通过。此写法与官方 `batch_norm_v3` 的 `VECOUT + CopyOutMeanOrVar` 一致。
2. **仿真通过≠真机通过（再次验证）**：CPU 仿真 140/140 全 PASS，但上述 `save_mean` 丢失只在真机出现（仿真下 `SetValue` 正常）。功能/精度以真机为准。
3. **占位 UT 陷阱（预期失败）**：`op_host` UT 的期望值 `"0 1 0 "`、`{0}` 为空实现占位；`op_kernel` UT 硬编码 `{totalNum=32, blockFactor=32, ubFactor=32}`，按本算子打包语义解码为 `DHW=32, C=0→1, N=0→1`，且 `epsilon` 位模式为极小 denormal（≈4.5e-44），使 `coefA≈−coefB≈4.7e21`、`y=1·coefA+coefB=0`——两条 UT 的精度比对失败均为占位值所致，任何正确实现都无法匹配（与 Trace / BNTrainingUpdate 同类）。
4. **fp16 golden 必须基于量化后的 half 输入**：独立校验程序的 golden 需用 `half→float` 量化后的 x，否则造成假失败；同时验证程序自身的 half 转换需正确处理 subnormal（早期实现移位越界产生 UB，一度误判 kernel 有 bug）。
5. **设备端标量超参**：`scale/offset` 恒为 float32（即便 x 为 fp16）；`GetValue` 读 GM 标量、`ReduceSum` 后需 `HardEvent::V_S` 同步再 `GetValue`。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| OP 包编译 + 打包（`bash build.sh`，ascend910b，half/float 双 kernel） | **PASS**（产出 `custom_opp_ubuntu_aarch64.run`） |
| **真机 NPU 批量验证**（aclnn 两段接口 + host golden 比对，17 shape × fp16/fp32 = 34 用例 × 4 轮） | **PASS 34/34 × 4**（稳定，无复现性抖动） |
| 真机覆盖 | (1,4,2,2,2) / (2,3,4,5,6) / (1,8,1,1,8) / (2,16,3,7,5) / (1,1,64,64,1) / (3,4,16,16,16) / (1,2,2,64,64) / (1,7,1,1,1) / (2,5,2,3,4) / (1,16,1,1,1) / (4,3,8,8,8) / (2,2,2,100,100) / (1,3,5000,1,1) / (1,64,1,1,1) / (8,32,4,4,4) / (1,4096,1,1,1) / (2,1,3,5,7)；含多核、C=4096、dhw=1、dhw>4096 跨块、非整除形状、fp16/fp32 |
| CPU 仿真独立校验（tikicpulib，自建 tiling，多形状 × 4 随机种子 + 多核） | **PASS 140/140**（fp32 最大绝对误差 ~7e-7，fp16 ≤ 1 ULP） |
| op_kernel UT | **编译、运行 OK**；精度比对 FAIL（预期：硬编码占位 tilingData 无法解码真实 shape，且 epsilon 为 denormal） |
| op_host UT | **FAIL（预期）**：期望值 `"0 1 0 "`、`{0}` 为空实现占位（totalNum=0 表示不处理任何元素），任何真实实现均无法匹配；`expectTilingKey=1` 与实现一致 |

说明：本地 UT 失败项均由模板占位值引起，不代表实现有误；权威验证为真机 NPU 批量精度比对（第六节前两行）。

## 七、性能要点

- **多核按通道并行**：通道内统计互相独立，`blockDim=min(coreNum, C)`，无原子、无跨核依赖、负载均衡；
- **数据通路全向量化**：统计用向量累加（跨段复用累加器）+ 每通道一次 `ReduceSum`，归一化用 `Muls/Adds` 广播系数，避免逐元素标量搬运；
- **双缓冲流水**：`TQue` 深度 2，MTE2 搬运与 Vector 计算重叠；
- **UB 占用恒定**：单块固定 4096 元素（fp32 约 128KB、fp16 约 112KB），统计量缓冲极小（按通道均分后每核仅若干元素），远低于 192KB UB 上限；
- **输出写出经 MTE3**：统计量与 y 均走 DMA 写出，由末尾 `MTE3_MTE2` 等待保证 kernel 退出前刷出。