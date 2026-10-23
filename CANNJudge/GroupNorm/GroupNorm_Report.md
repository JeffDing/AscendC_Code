# GroupNorm 算子实现报告

## 一、任务概述

在昇腾 NPU（Ascend C，910B3）上实现 GroupNorm 算子：输入 `x(N, C, *)`、`gamma(C,)`、`beta(C,)`，属性 `num_groups=G`、`epsilon`（默认 1e-5）。对每个样本 `n` 与每组 `g`，在该组 `C/G` 个通道 × 空间维（`hwSize`，C 之后所有维度乘积）的元素集合上计算 `mean/var`，输出 `y = (x-mean)/sqrt(var+eps)*gamma[c] + beta[c]`。`x/y` 支持 fp16/fp32（内部 FP32 计算），`gamma/beta` 恒为 fp32，输出 `y` 与 `x` 同 shape 同 dtype。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/group_norm.h` | 实现 kernel 全部逻辑（6 处 TODO，注释均保留） |
| `op_host/group_norm_tiling.cpp` | 实现 tiling 计算（2 处 TODO，注释保留） |
| `op_host/group_norm_infershape.cpp` | 输出 y 与 x 同 shape（1 处 TODO，注释保留） |

**未修改任何其他文件**：`tests/` 目录全部文件、`op_kernel/group_norm_tiling_data.h`（3 字段结构）、`group_norm.cpp`、`group_norm_tiling_key.h`、`group_norm_def.cpp`、构建脚本均保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`GroupNormTilingData` 保持原有 3 个 int64 字段，host 与 kernel 两侧位域约定严格一致：

| 字段 | 位域 | 语义 |
|---|---|---|
| `blockFactor` | bit[0:32) | `hwSize`（x 中 C 之后所有维度的乘积） |
| | bit[32:48) | `C`（通道数） |
| | bit[48:64) | `G`（num_groups 分组数） |
| `ubFactor` | bit[0:32) | `epsilon` 的 IEEE-754 位模式 |
| | bit[32:64) | `N`（batch 数） |
| `totalNum` | 保留 | `N*C*hwSize`（kernel 由位域字段推导，不依赖该字段） |

- kernel 端 UB 单块元素数固定为编译期常量 4096（fp32 占 UB 约 128KB、fp16 约 112KB，均低于 192KB 上限），无需 host 下发；
- kernel UT 的占位赋值 `{16,16,16}` 在此语义下解码出 `hwSize=16, C→1, G→1, eps≈2.2e-44(denormal), N→1`，属占位值陷阱（见第六节说明），**无需修改 test 文件**。

## 四、算法设计

### 4.1 关键观察：组内元素在内存中连续

x 为 (N, C, hwSize) 连续布局时，组 `gIdx = n*G + g` 覆盖通道 `[g*C/G, (g+1)*C/G)`，这些通道在内存中首尾相接，故**每组恰为一段连续区间 `[gIdx*groupSize, +groupSize)`**（`groupSize = C/G * hwSize`）。组级归约因此对单段连续内存做两遍扫描即可，无需任何跨步 gather。

### 4.2 kernel 侧（group_norm.h）

多核切分：按组均分（`N*G` 个组，组间归约互相独立，天然无跨核依赖），每核负责连续组区间。每组两阶段：

1. **pass 1 统计**：按 4096 元素分块 `DataCopyPad` 搬入（inQ 深度 2 double-buffer），fp16 先 `Cast` 升 fp32；`acc[p%blk]+=x`、`accSq[p%blk]+=x*x` 位置模累加（跨块不清零），整组扫描完一次 `ReduceSum` 得 `sum/sumsq`；`mean=sum/size`、`var=sumsq/size-mean²`（负值钳 0），`invStd=1/sqrt(var+eps)`。
2. **pass 2 归一化+仿射**：gamma/beta 是**通道级**的（同组内各通道系数不同），故按组内通道循环，每通道标量读 `gammaGM.GetValue(c)/betaGM.GetValue(c)` 得 `coefA=invStd*gamma`、`coefB=beta-mean*coefA`，对该通道 hwSize 个连续元素 `Muls+Adds`（fp16 路径 fp32 计算后 `Cast RINT` 写回），经 outQ + `DataCopyPad` 写出。

同步要点：
- `ReduceSum` 写回后先 `PipeBarrier<PIPE_V>` drain，再 `SetFlag/WaitFlag<HardEvent::V_S>` 取标量（reduce 类指令写回必须先屏障再读取的已验证规则）；
- kernel 退出前 `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>` 确保全部 GM 写出完成；
- 唯一输出 y 全部走 `DataCopyPad`/MTE3 通路，无多核 `GlobalTensor::SetValue` 直写 GM（已知的丢数据陷阱），输出间互不重叠。

### 4.3 host 侧（group_norm_tiling.cpp）

- 解析 x 的 (N, C, *)（兼容低维退化），并用 gamma 的 shape 校正 C；属性取 `num_groups`（必选）与 `epsilon`（可选，默认 1e-5，非正/NaN 回退默认）；
- 多核：`blockDim = min(coreNumAiv, N*G)`；
- workspace = 0；
- tilingKey：fp16/bf16 → MODE_0（half kernel），其余 → MODE_1（float kernel）；
- 按第三节位域打包 TilingData。

### 4.4 infershape（group_norm_infershape.cpp）

`y` 与 `x` 同 shape/同 dtype，直接 `*output_shape = *input_shape`。

## 五、关键决策依据（实验实证）

1. **fp16 精度"假失败"根因在验证程序而非 kernel**：初版仿真 fp16 大形状（C≥32 且 hwSize>1）报数百个超差元素（个别 y=8.2 级偏差）。聚焦实验显示错误完全确定（单核=多核），且 fp16 精确可表示数据/全 1 数据 0 错误。定位为**验证程序自身 `HalfToFloat` 的 subnormal 规范化公式错误**（`e+132` 应为 `e+114`），把 fp16 subnormal 输入放大 2^18 倍（3.1e-5→8.2），污染 golden 的组统计量并扩散成全组偏差；kernel 对该元素的输出经反推完全正确。修复验证程序后 fp16 全数通过（与同项目 BatchNorm3D 的已知陷阱一致：fp16 失败先审验证程序 half 转换）。
2. **组内存连续性**：利用 (N,C,*) 布局下组区间连续的性质，pass 1 归约无需跨步访问，pass 2 仅按通道切分以适配通道级 gamma/beta，拷贝/计算全部 32B 对齐友好。
3. **归约取标量必须屏障**：`ReduceSum` 写回后 `PipeBarrier<PIPE_V>` + V_S 事件，避免 V/S 跨管线读竞争（FlashAttention 已验证规则）；该屏障落在 V 管线本就必须等待的点，性能零回退。
4. **占位 tiling 防护**：kernel 对 `hwSize/C/G/N<=0`、`eps` 非正或 NaN、`C/G<=0` 全部回退安全值，避免除零与死循环。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 独立 CPU 仿真（正确 tiling，double golden 比对）：14 个 fp32 + 11 个 fp16 形状，覆盖单核/多核（≤40 核）、hwSize=1、每组 1 通道（G=C）、G=1、跨块（groupSize=8192>4096）、非整除、C=512 | **25/25 PASS**（fp32 最大绝对误差 ~2e-6；fp16 ≤1 ULP，个别 2 ULP 内） |
| 真机 NPU 910B3 aclnn 批量验证：25 个形状（含 16×512×7×7、G=32），fp16/fp32 | **25/25 × 4 轮全部 PASS**（fp32 ~1e-6；fp16 ≤3 ULP） |
| 算子包安装一致性 | build 产物与 `opp/vendors/group_norm_custom` 安装件 md5 逐一核对一致；`load_priority` 已置前 |
| op_kernel UT | **PASSED**（kernel 正常执行无异常） |
| kernel UT 精度比对（compare_data.py） | **FAILED（预期内）**：UT 硬编码占位 tiling `{16,16,16}` 解码出 denormal eps≈2.2e-44，`x≡1` 时 `coefA≈-coefB≈6.7e21` 致 `y=1*coefA+coefB=0`，任何正确实现均无法匹配该占位 golden（与 BatchNorm3D/BNTrainingUpdate 占位 UT 同类）；同形状正确 tiling 下仿真/真机均 PASS |
| op_host UT | **FAILED（预期内）**：期望值 `"0 1 0 "` 是空实现占位；实际输出 `"16 562967133290500 5220320684 "` 解码为 `totalNum=16, hwSize=4, C=4, G=2, eps=1e-5, N=1`，与 (1,4,2,2)+G=2 输入完全一致；tilingKey=1（fp32→MODE_1）匹配通过 |

说明：真机验证前已确认 NPU 无残留进程、安装 kernel 与 build 产物 md5 一致（避免陈旧 kernel 误导）。

## 七、性能要点

- 组区间连续 → 统计与归一化均为纯顺序 DMA（`DataCopyPad`），无跨步 gather；
- inQ/outQ 深度 2 double-buffer，MTE2 搬运与 Vector 计算流水重叠；
- 位置模累加（`acc[p%blk]`）使跨块累加零额外指令，整组一次 `ReduceSum`；
- 归一化融合为 `Muls+Adds` 两条向量指令（`y=x*A+B` 系数预合并），fp16 仅首尾各一次 Cast；
- 多核按组均分、无跨核依赖，`blockDim=min(coreNum, N*G)` 无空转核；单核 UB 占用 fp32≈128KB / fp16≈112KB，低于 192KB 上限。
