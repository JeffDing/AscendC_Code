# GroupNormSiluQuant 算子实现报告

## 一、任务概述

在昇腾 NPU（Ascend C）上实现 GroupNormSiluQuant 融合算子，对输入 `self(N, C, *)`（fp16/fp32）依次完成：

1. **GroupNorm**：按 `group` 属性将 C 通道分组，组内（C/G 个通道 × C 之后所有维度）计算 mean / rstd（`rstd = 1/sqrt(var+eps)`），归一化后做通道级仿射 `gn = ((x-mean)*rstd)*gamma + beta`（gamma/beta (C,)，与 self 同 dtype）；
2. **SiLU**（可选，属性 `activateSilu`，默认开启）：`act = gn / (1 + exp(-gn))`；
3. **量化**：`out = rint(act / quantScale)` clip 到 [-128, 127]，输出 int8（quantScale 为 fp32 标量或 (C,)）。

三个输出：`out` 与 self 同 shape（int8）；`meanOut` / `rstdOut` 为 (N, group)，dtype 与 self 一致。组内元素在 (N, C, *) 连续布局下恰为连续区间，多核按组（N*G 个）均分，组间无依赖。

评测要求 int8 输出比对零容差（bitwise 一致），为此对 golden 的 numpy 浮点求和路径做了完整逆向并逐位复刻。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/group_norm_silu_quant.h` | 实现 kernel 全部逻辑（6 处 TODO，注释均保留） |
| `op_host/group_norm_silu_quant_tiling.cpp` | 实现 tiling 计算（2 处 TODO，注释保留） |
| `op_host/group_norm_silu_quant_infershape.cpp` | 实现三输出形状推导（1 处 TODO，注释保留） |

**未修改任何其他文件**：`tests/` 全部文件、`op_kernel/group_norm_silu_quant.cpp`、`group_norm_silu_quant_tiling_data.h`、`group_norm_silu_quant_tiling_key.h`、`group_norm_silu_quant_def.cpp`、`CMakeLists.txt`、`build.sh` 等均保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`GroupNormSiluQuantTilingData` 保持原有 3 个 int64 字段，位域约定如下（host 与 kernel 两侧严格一致）：

| 字段 | 位域 | 语义 |
|---|---|---|
| `blockFactor` | [0:32) | hwSize（C 之后所有维度乘积） |
| | [32:48) | C（通道数） |
| | [48:64) | G（num_groups） |
| `ubFactor` | [0:32) | epsilon 的 IEEE-754 位模式 |
| | [32:62) | N（batch 数） |
| | [62] | activateSilu（1=开启 SiLU） |
| | [63] | quantScale 是否按通道（1=(C,)，0=标量） |
| `totalNum` | 保留 | = N*C*hwSize，kernel 由位域字段自行推导，**不读取该字段**（规避真机 TilingData 首字段读取不可靠的已知风险） |

- kernel UT 的占位 tiling `{16,16,16}` 在此语义下解码为 hwSize=16/C=1/G=1/eps≈2.2e-44(denormal)/N=1/silu=0/qs 标量，kernel 内有完备的退化保护，不会挂死或越界；
- UB 单块元素数 4096、统计量收集容量 4096 均为编译期常量（fp32 约 137KB / fp16 约 121KB，均 < 192KB 池上限）。

## 四、算法设计

### 4.1 numpy f32 求和算法的完整逆向（M3 模型，本算子正确性的基石）

评测 golden 的 mean/var 由 `np.sum`（numpy 1.26.4 aarch64）计算，其 f32 求和路径经黑盒探测**完整恢复**（多 seed 数值验证全规模 0 mismatch）：

```
sum(a, n):
  n < 8        : 标量顺序和
  8 ≤ n ≤ 128  : 8-lane（lane = idx%8 顺序累加 16 项跨距 8）
                 + 横向平衡树 ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
                 + 尾巴（n%8 ≠ 0 时在横向结果上顺序补加）
  128 < n ≤ 8192: 二分递归（n2 = n/2 向下对齐 8）
  n > 8192     : 8192 定长段（尾段可短），段间顺序合并 acc = f32(acc + 段和)
```

探测方法（对后续同类算子可复用）：结合律探针（2^24 大数吸收法）恢复加法树分割边界；半 ulp 探针（2^-24/2^-23 组合）区分横向树形状与累加顺序；块和注入实验区分块间组合结构；hand_tree 分解检验定位顶层合并方式。关键转折：n=32768 起顶层不再是二分而是「4 段 8192 顺序合并」，进而在 65536+ 规模确立「8192 定长段」结构。

### 4.2 kernel 侧 M3 逐位复刻（group_norm_silu_quant.h）

每组两遍求和均按 M3 结构：

- **叶（≤128 元素）**：CopyIn → fp16 经 Cast 升 f32 →（pass1b：Sub mean / Mul 自乘）→ **8-lane 行累加**（16 次向量 `Add(acc8, acc8, f[8k], 8)`，数值等价于 numpy 的跨距 8 顺序累加）→ **`ReduceSum(8)` 横向**——真机实验确证其硬件归约树恰为平衡树，与 numpy 横向逐位一致 → 尾巴元素标量补加；
- **满段 / 2 的幂区间（≤8192）**：叶按 **bitrev 序**求和写入 leafVal，再做**对折折叠**（`Add(leafVal, leafVal, leafVal[h], h)`，h≥8 向量层；尾部 ≤8 元素标量对折）——bitrev+对折在数学上与「二分递归平衡树」逐位等价，从而把树形合并完全向量化；
- **尾段（非 2 的幂 ≤8191）**：显式栈迭代模拟二分递归（区间栈 + 值栈，S 域标量执行），叶序与递归完全一致；
- **段间**：顺序标量合并；
- **统计量标量链与 golden 严格同构**：`mean = sum / n`（**除法**，与 np.mean 的 true_divide 舍入一致，不用乘 1/n）、`var = sqsum / n`、`invStd = 1/sqrt(var+eps)`；
- **两遍法**：pass1a 求 sum → mean；pass1b 对 `(x-mean)^2` 走同一 M3 结构求 sqsum → var。规避单遍 `E[x²]-mean²` 的灾难性消解（实测大 groupSize + 大均值下 var 相对误差可达 0.46，直接导致 rstd 超阈——这是初版评测 11-15 WA 的主因之一）。

### 4.3 f16 subnormal 输入清洗（FTZ）

硬件 vconv 对 **f16 subnormal 输入产生 NaN**（实测 Cast 后 sum 全 NaN；正态分布数据约 5e-5 概率落入 subnormal，8M 元素用例必现数百个，导致 fp16 大用例全错）。在所有 Cast 前用 `Abs + Compares + Select` 将 `|x| < 2^-14` 的元素置 0：统计量误差 ~1e-9 量级可忽略；subnormal 元素自身的 gn 偏差 ~rstd×1e-5，量化后基本不变。

**关键陷阱**：`Compares/Select` 的 count 需 256 字节对齐（half 即 128 元素），count=8 的未对齐调用产生坏 mask 把整组数据清零（c01 类小组全部 mean=0）。修复：count 向上对齐到 128 的倍数，配合 Duplicate 清零 xa 越界段、Select 越界写仅触碰本块 buffer 内区域；len<8 的标量路径直接跳过（subnormal 概率可忽略）。

### 4.4 量化 Cast 链路

`f32 → s32(CAST_RINT) → f32(CAST_NONE) → f16(CAST_NONE) → s8(CAST_RINT)` 四步：仅 f32→s32 一次 RNE 舍入（与 np.rint 的 half-to-even 一致），整数在 [-128,127] 内经 f32/f16 均精确可表示；同时避开 f32→f16→s8 的双舍入与 s32→f16 直转（仿真器该路径输出 ±0）。gn 严格按 golden 左结合顺序（`Adds(-mean)→Muls(invStd)→Muls(gamma)→Adds(beta)`），除法一律用向量 `Div`。

### 4.5 统计量写出

mean/rstd 先 `SetValue` 到 UB 收集缓冲（容量 4096 组），攒满或收尾时 `FlushStats()` 经 MTE3 DataCopyPad 写出（多核 `GlobalTensor::SetValue` 直写 GM 会大面积丢失，BatchNorm3D 已确证）；写出偏移 `statFlushBase_` 初始化为 `groupStart_`（(N,G) 连续布局下本核组区间即输出区间）；fp16 时 flush 链 S_V→Cast f32→f16(RINT)→V_MTE3→DataCopyPad，fp32 直接 S_MTE3→DataCopyPad，末尾 MTE3_S 等待后再复用缓冲，支持任意组数的多轮 flush。

### 4.6 host 侧

- 解析 self 的 (N, C, *)（兼容退化维度），gamma shape 校正 C；quantScale 元素数 >1 判定按通道；
- 属性：group（必选 idx0，截断 [1,C]）、eps（可选 idx1）、activateSilu（可选 idx2）；
- blockDim = min(coreNum, N*G)，组间无依赖；workspace=0；位域打包三字段；tilingKey：fp16/bf16→MODE_0，fp32→MODE_1。

### 4.7 infershape

`out` = self shape；`meanOut`/`rstdOut` = (N, group)。

## 五、关键决策依据（实验实证）

1. **评测 11-15 WA 的根因链**（用户反馈驱动）：WA 时「输出错误占比 0.00%」+ 本地复现实验表明：① 单遍 var 公式消解使 rstd 相对误差达 3e-4~0.46（超阈）；② 求和树与 numpy 不同导致 mean 差 1-5 ulp，rint 边界翻转率 ~1e-4。两项均已修复：两遍法 + M3 逐位复刻后 mean/rstd **全 dtype 全规模 bitwise 一致**。
2. **ReduceSum(8) 硬件树 = 平衡树**：特构 8-lane 探针（1 与 2^-24 阵列）真机验证 3f800003（平衡）而非对折/顺序值；随机数据分级探针（128→131072）全 0 ulp 交叉确认。
3. **bitrev+对折 = 平衡二分树**：数学等价（对折树的输入按 bitrev 置换后相邻对配对结构重排为平衡树），配合「向量层 h≥8（Add 32B 对齐约束）+ 尾部 ≤8 元素标量对折」在真机全规模验证。
4. **Add/Compares/Select 的 32B/256B 对齐铁律**：`leafVal[h]` 子 tensor 在 h<8 时地址失配（实测报错）；Compares/Select count 未 256B 对齐时静默产生坏 mask（实测小组数据被清零）。
5. **增量构建不追踪 .h 依赖**：中途多轮 A/B 实验全部无效（二进制从未更新），全量重编后才暴露真实行为——调试期务必 `rm -rf build` 后验证。
6. **仿真器陷阱**：s32→f16 CAST_NONE 输出 ±0（绕道 f32 中转）；f16 subnormal 的 Cast 仿真输出错值、真机 NaN（FTZ 规避）。
7. **TilingData 首字段不读**；**统计量多核写出走 UB 收集 + MTE3**（继承 BatchNorm3D/GroupNorm 经验）。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| CPU 仿真探针（tikicpulib）：32 个形状/分布用例 × fp16/fp32，覆盖 hwSize=1、G=C、G=1、3D、跨块、C=512、多核、silu 开/关、quantScale 标量/按通道、大 eps、常数、大均值消解复现组（c15/c16） | **全部 PASS**：int8 全部 bitwise 一致 |
| 真机 NPU 910B3 aclnn 全量验证（90 用例含树序/lane 通道/分级/行累加/subnormal 专项探针） | **88/90 × 3 轮稳定 PASS**：fp16 与 fp32 全部正式用例 int8 bitwise 一致、mean/rstd bitwise 一致。剩余：c20 为验证脚本自身的 meta 笔误（f16 数据误标 fp32，非 kernel 问题）；fp32 c15 存在 **1/8388608** 单点翻转（act 距 rint 边界 2.2 ulp，源自硬件 vexp 与 np.exp 的 1 ulp 差异） |
| numpy 求和算法逆向（M3 模型） | python 端全规模（8193~2097152）× 8 seed **0 mismatch** |
| op_kernel UT + compare_data.py | int8 输出 **bitwise PASS**；mean/rstd 与 op_host UT 失败为占位期望（模板空实现占位符，任何真实实现均无法匹配，与 GroupNorm 先例一致，不回滚） |

说明：功能/精度验证以真机结果为准。fp32 c15 的单点翻转若评测仍判 WA，下一步是对 numpy f32 exp 多项式的逐位复刻（np.exp 实测非正确舍入，需逆向其 SIMD 实现）。

## 七、性能要点

- 组内存连续 + 多核按组均分（blockDim = min(coreNum, N*G)），无跨核依赖；
- 满段路径 bitrev+对折把树形合并完全向量化（64 叶仅 1 次 GetValue），尾段显式栈仅每核最多一次；
- inQ double-buffer；MTE2 与 V 流水重叠；quantScale 标量情形仅读一次 GM；
- 统计量 UB 收集 + MTE3 集中写出；
- FTZ 清洗仅增加 3 条向量指令/块；UB 总占用 fp32≈137KB / fp16≈121KB（<192KB 上限）。
