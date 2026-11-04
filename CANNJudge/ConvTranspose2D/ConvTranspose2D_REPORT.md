# ConvTranspose2D 算子实现报告

## 一、任务概述

在昇腾 NPU（Ascend C）上实现 ConvTranspose2D 算子：标准转置卷积（反卷积，groups=1）。

- **输入**：`x` (N, C_in, H_in, W_in)、`w` (C_in, C_out, kH, kW)、`b` (C_out,) 可选偏置
- **输出**：`y` (N, C_out, H_out, W_out)，dtype 与 `x` 一致（fp16/fp32，内部 FP32 累加）
- **尺寸公式**：`H_out = (H_in-1)*stride - 2*padding + dilation*(kH-1) + output_padding + 1`（W 同理）
- **语义**：`y[n,co,oh,ow] = b[co] + Σ_{kh,kw,ci} x[n,ci,ih,iw] * w[ci,co,kh,kw]`，其中 `ih=(oh+padding-kh*dilation)/stride`、`iw=(ow+padding-kw*dilation)/stride`，仅当 ih/iw 为整数且落在输入范围内时该项计入
- 支持 stride / padding / output_padding / dilation / kH≠kW / 无偏置 / 非对齐 H,W / 退化场景；dtype 分 fp16（schMode 0）与 fp32（schMode 1）两路 tilingKey

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/conv_transpose2_d.h` | 实现 kernel 全部逻辑（7 处 TODO，注释均保留） |
| `op_host/conv_transpose2_d_tiling.cpp` | 实现 tiling 计算（2 处 TODO，注释保留） |
| `op_host/conv_transpose2_d_infershape.cpp` | 实现输出 shape 推导（1 处 TODO，注释保留） |

**未修改任何其他文件**：`tests/` 目录全部文件、`op_kernel/conv_transpose2_d_tiling_data.h`（无 TODO）、`conv_transpose2_d.cpp`、`conv_transpose2_d_tiling_key.h`、`build.sh` 均保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`ConvTranspose2DTilingData` 保持原有 3 个字段。因转置卷积需要传递 N/C_in/C_out/H_in/W_in/kH/kW/stride/padding/output_padding/dilation/hasBias 共 12 个参数，远超 3 字段容量，采用 **64 位位域打包**方案：

| 字段 | 语义 |
|---|---|
| `totalNum` | 冗余保存 `N*C_out`（总任务数）。**kernel 刻意不读**：真实框架下 TilingData 首字段在 kernel 侧读取不可靠（偶发读到 0/8，已由常量 tiling 受控实验确证），故仅作冗余信息保留 |
| `blockFactor` | bit[0:13)=N，bit[13:26)=C_in，bit[26:39)=C_out，bit[39:50)=H_in，bit[50:61)=W_in |
| `ubFactor` | bit[0:4)=kH，bit[4:8)=kW，bit[8:20)=stride，bit[20:32)=padding，bit[32:44)=output_padding，bit[44:56)=dilation，bit[56:57)=hasBias |

- host 侧对每个字段做 **位域宽度上界检查**，超限时 `GRAPH_FAILED`，避免静默截断导致 kernel 侧解码错位；
- kernel 侧对解码结果做 **退化保护**（<1 的维度/步长抬到 1、hOut/wOut<1 抬到 1），保证非法/占位 tiling 下不除零、不死循环、不越界。

## 四、算法设计

### 4.1 kernel 侧（conv_transpose2_d.h）

采用转置卷积的 **聚合（gather）形式**，任务 = 一个 `(n, co)` 输出平面 `(H_out × W_out)`，任务间无数据依赖。

**关键重排：kw 提到最外层。** 对固定 `(oh, kh, kw)`，能使 `iw` 为整数的 `ow` 构成唯一残差类：

```
r_kw = (kw*dilation - padding) mod stride，  ow = owLo + t*stride，
owLo = max(r_kw, kw*dilation - padding)，  iw0 = (owLo + padding - kw*dilation)/stride ≥ 0
```

于是「输入列 iw」与「输出列 ow」双双等差连续：

- **输入侧**：`x[n,ci,ih,iw0..iw0+cnt)` 是一段连续内存，单条 `DataCopyPad` 取回；
- **计算侧**：每个 `(kh,kw,ci)` 一次 `Axpy` 外积（`acc += xRow * wScalar`），全程向量化；
- **输出侧**：`acc` 的 offset 恒为 0（规避真机上「dst 带运行期 offset 的向量指令只有 offset=0 生效」的缺陷），算完立即标量展开进 `outFBuf_`（连续输出列 float 缓冲），最后整段一次性 `DataCopyPad` 写 GM。

**ow 分块**：单块残差类个数 `trBlk = ceil(remain/stride)` 受 `TR_MAX=4096` 与 UB 容量共同约束，`owStart += owBlk` 循环覆盖超宽输出。块内首列 `first = max(owStart + r_kw, owLo)`，输入起点随块前移 `iw0 += (first-owLo)/stride`，块内个数上界同时受输入列范围 `kwCntMax[kw]-iwShift` 约束。

**权重段加载**：`w[:, co, kh, kw]` 沿 C_in 跨步 `wPlane_`，用 `DataCopyPad`（blockCount=C_in, blockLen=sizeof(T), srcStride=(wPlane_-1)*sizeof(T)）一次 gather 到 `wSegBuf_`。`C_in > CIN_MAX=8192` 时回退为逐元素 GM 读 `w`。wSeg 每行仅 1 个有效元素，但 `DataCopyPad` 按 32B 块计费，故 wSegBuf_ 按「行数 × 每行 32B 可容纳元素数」分配。

**UB 分配**：`trBlkAl`/`owBlkAl`/`wSegElems` 均做 32B 对齐，且下限 `kMinVecElems = 64/sizeof(T)`——CPU 调试构建的向量指令要求 dst/src 张量至少 64B（`CheckFunVecBinaryScalarDiffType`），小 shape 时不抬下限会 abort。

### 4.2 host 侧（conv_transpose2_d_tiling.cpp）

- 解析 shape 与 4 个属性（兼容「标量 int」与「list int」两种下发形式）；
- 合法性校验（维度 ≥1、stride/dilation ≥1、padding/output_padding ≥0、输出尺寸 >0、位域宽度上界）；
- 多核切分：`totalTask = N*C_out`，`usedCore = min(coreNum, totalTask)`，`SetBlockDim(usedCore)`，任务按连续区间均分，无跨核竞争、无需跨核同步；**不使用 workspace**（`WS_SYS_SIZE = 0`）；
- tilingKey：fp16/bf16 → `MODE_0`（half kernel），其余 → `MODE_1`（float kernel）。

### 4.3 infershape（conv_transpose2_d_infershape.cpp）

按标准公式推导输出 shape `(N, C_out, H_out, W_out)`，非 4 维退化输入按 `DimAt(..., fallback)` 兼容，输出尺寸非正时 `GRAPH_FAILED`。

## 五、关键决策依据（实验实证）

1. **不用 `SetValue` 写 GM**：多核各自标量写 GM 会大面积丢失（已确证教训），改为 UB 收集 + 单条 MTE3 `DataCopyPad` 整段写出。
2. **不用 `HardEvent::MTE2_S`**：该事件对在平台上不可靠（过评测算子 0 处使用），改用 `FetchEventID` + `SetFlag`/`WaitFlag` 的 `MTE2_V` / `V_MTE3` / `MTE3_MTE2` 显式同步。
3. **不用 `PipeBarrier<PIPE_ALL>`**：单核下会 hang，用细粒度事件对替代。
4. **TilingData 首字段不可靠**：真实框架下 `totalNum` 偶发读到 0/8（常量 tiling 受控实验确证），故 kernel 只从 `blockFactor`/`ubFactor` 位域解包，`totalNum` 仅冗余。
5. **`Cast<float,float>`（CAST_NONE 同型）在 dav_c220 直接 abort**：同型走 UB→UB `DataCopy`（`CastOrCopy<T>()` 模板按 `std::is_same` 分支）。
6. **向量指令 64B 下限**：CPU 调试构建下 dst/src 不足 64B 的向量指令 abort，故对齐后统一下限 `kMinVecElems`。
7. **wSeg 32B 计费**：`DataCopyPad` 每行按 32B 块计费，wSegBuf_ 须按「行数 × 32B/sizeof(T)」分配，否则越界。
8. **`DataCopyExtParams` 与 `DataCopyParams` 单位不同**：前者 blockLen 单位为字节，后者单位为 32B 块，混用会错位。
9. **ow 分块的输入列随块前移**（本轮修复的真实缺陷）：`kwOwLo[]`/`kwIw0[]` 是全局首列，`owLo < owStart` 时 `owOff` 为负、`iw0` 未随块前移、`trOut` 把 owLo 之前的列算进块内计数。正常 shape 下（wIn ≤ 2047 位域限制 → trMax ≤ 2062 < TR_MAX=4096）该路径不可达，但通过覆盖配置强制触发后 28/28 FAIL 暴露了缺陷，已按 `first/iwShift/cnt` 重算修复。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| CPU 仿真探针（题面示例 + 对抗形状 31 用例：dilation/padding/output_padding/stride>2/kH≠kW/无偏置/C_in≠C_out/非对齐/退化/多核/大 W/fp16） | **31/31 PASS**（abs+rel 双阈值：fp32 1e-4 / fp16 1e-3，double 精度独立 golden） |
| 覆盖配置加压（`TR_MAX=1/2/3`、`CIN_MAX=1` 四种配置 × 31 用例 = 124 项） | **124/124 PASS**（强制触发 ow 分块循环与 wSeg 回退路径；无宏时与交付 kernel 行为逐例一致） |
| 平台全量构建（`build.sh -j8`，CANN 9.0.0 / ascend910b） | **成功**，两个 tiling key 二进制与 `custom_opp_ubuntu_aarch64.run` 均生成 |
| op_kernel UT（真机 910B3，`ConvTranspose2DKernelTest.test_kernel_run`） | **PASSED**（5406ms） |
| 真机 910B3 题面 shape + 真实 tiling 编码闭环验证 | **逐位一致**（MERE=0.000000, MARE=0.000000, bitwise equal，与官方 `float32_golden_conv_transpose2_d.bin` 比对） |
| op_host UT | **FAIL（预期内）**：期望值 `"0 1 0"` 是空实现占位常量，实际输出 `"4 9011597569720321 72075186223972659"` 是真实 tiling 编码（totalNum=4=N×C_out），与参考实现一致，非缺陷 |

**关于 UT 精度比对的说明**：`tests/ut/op_kernel/test_conv_transpose2_d.cpp` 的 tilingData 使用初始模板占位值 `totalNum=blockFactor=ubFactor=256`（该文件不可修改，且 TODO 注释明确要求「修改 TilingData 结构后请更新字段名和赋值」）。kernel 按位域解码后得到 `N=256, C_in=1, C_out=1, H_in=1, W_in=1, kH=1, kW=1, stride=1`——合法但退化的 shape，每个任务仅写出 1 个偏置值（256 个非零元素全为偏置 1），与按真实 shape `1×4×8×8 / 4×4×3×3` 计算的 golden 不可比。故该 compare 失败是**脚手架占位 tiling 与 golden 语义不匹配**所致，与 kernel 正确性无关。为闭环该缺口，另建独立验证程序（不改 UT、不改交付文件），用 `op_host` 的位域编码为 UT 的真实 shape 生成 tilingData 并驱动交付 kernel，在真机 910B3 上与官方 golden **逐位一致**（见上表）。

## 七、性能要点

- **聚合形式 + kw 最外层重排**：输入列/输出列双双等差连续，`DataCopyPad` 单条 DMA 取回整段连续输入，`Axpy` 全程向量化，无逐元素标量 GM 读；
- **权重段一次 gather**：每个 `(kh,kw)` 仅一次 `DataCopyPad` 取回整段 `w[:,co,kh,kw]`，摊薄到 C_in 个外积上；
- **输出整段一次 MTE3**：块内连续输出列在 `outFBuf_` 中累积，单条 `DataCopyPad` 写出；
- **多核按任务均分**：任务 = `(n, co)` 输出平面，无数据依赖，核数 = `min(coreNum, N*C_out)` 无空转；
- **UB 占用**：`trBlkAl×(2×sizeof(float) + sizeof(T)) + owBlkAl×(sizeof(float)+sizeof(T)) + wSegElems×sizeof(T)`，峰值远低于 192KB 池上限；
- **解析法性能模型**（按 kernel 循环结构精确统计指令条数，130ns/条保守标定）：11 个代表性 shape（含 1×1024×8×8 / 1024×1024×3×3 极端用例）8 核并行最坏 **0.46s vs 平台硬超时 3.00s，余量 6.5 倍**。

## 八、结论

ConvTranspose2D 算子已在 Ascend C 上完整实现：3 个白名单文件全部完成、TODO 注释全部保留、未触碰任何其他文件。正确性经 CPU 仿真（31 用例 + 124 项覆盖加压）、平台全量构建、真机 910B3 kernel UT 与题面 shape 真实 tiling 闭环（与官方 golden 逐位一致）四层验证；性能模型显示最坏 shape 余量 6.5 倍。交付物 `custom_opp_ubuntu_aarch64.run` 已生成，可提交平台评测。
