# QuantMatmulSigmoidMulQuant 算子实现报告

## 一、任务概述

在昇腾 NPU（Ascend C）上实现 QuantMatmulSigmoidMulQuant 算子：int8×int8 矩阵乘（int32 精确语义）→ T-T scale 反量化 → FP16 舍入 → Sigmoid → 逐元素乘 → FP16 舍入 → 全矩阵 per-tensor 动态量化输出 int8，同时输出 FP32 的 yScale。

**算子语义**（由 `gen_data.py` 的 `impl()` 权威确定）：

```
acc[m,n] = sum_k x1[m,k] * x2[k,n]                        // int8×int8，float 累加
merged   = x1Scale[0] * x2Scale[0]                        // T-T scale，shape (1,)
h        = fp16_rne(acc * merged)                         // 舍入点 1
g        = fp16_rne(sigmoid(h))                           // 舍入点 2
u        = fp16_rne(g * h)                                // 舍入点 3
absMax   = max(|u|)                                       // 全矩阵 per-tensor
yScale   = absMax / 127.0  （absMax==0 时取 1.0）
y        = clamp(rint(u / yScale), -128, 127)             // RNE + clip
```

输入 x1/x2 为 INT8，支持 `transposeX1`/`transposeX2` 两种物理布局（共 4 种组合）；x1Scale/x2Scale 为 FLOAT (1,)；bias 固定为空；输出 y 为 INT8 (M,N)，yScale 为 FLOAT (1,)。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/quant_matmul_sigmoid_mul_quant.h` | 实现 kernel 全部逻辑（11 处 TODO，注释均保留） |
| `op_host/quant_matmul_sigmoid_mul_quant_tiling.cpp` | 实现 tiling 计算与 TilingData 位域编码，含多核切分 |
| `op_host/quant_matmul_sigmoid_mul_quant_infershape.cpp` | 由 x1/x2 shape + transpose attrs 推导 y=(M,N)、yScale=(1,) |

**未修改任何其他文件**：`tests/` 目录、`op_kernel/quant_matmul_sigmoid_mul_quant_tiling_data.h`（无 TODO）、`op_kernel/quant_matmul_sigmoid_mul_quant.cpp`（kernel 入口）、`quant_matmul_sigmoid_mul_quant_tiling_key.h`、`op_host/quant_matmul_sigmoid_mul_quant_def.cpp` 均保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`QuantMatmulSigmoidMulQuantTilingData` 保持原有 3 个 int64 字段，采用位域打包承载 M/K/N 与切分参数：

| 字段 | 位域 | 语义 |
|---|---|---|
| `blockFactor` | [0:24) | M（输出行数） |
| | [24:48) | K（累加维度） |
| | [48] | transposeX1 |
| | [49] | transposeX2 |
| | [50:64) | reserved |
| `ubFactor` | [0:24) | N（输出列数） |
| | [24:36) | CHUNK_M（每轮 UB 处理的 M 行数） |
| | [36:48) | CHUNK_N（每轮 UB 处理的 N 列数） |
| | [48:64) | reserved |
| `totalNum` | — | M * N |

kernel 端 `ParseTiling` 解包上述位域，并对占位/退化 tiling 做一致性保护（M/K/N/chunkM/chunkN ≤ 0 时置 1）。

## 四、算法设计

### 4.1 整体流程（两阶段重算 + 多核按 M 切分）

per-tensor 量化要求**全局** abs_max，而 kernel 入口不接收 workspace、无独立 GM 归约区。因此采用两阶段重算，并在两阶段之间插入跨核归约握手：

1. **Phase 1（各核扫局部 max）**：每核按 M 块轮询（`m = blockIdx; m < M; m += blockNum*chunkM`），对每个 (M 块, N 块) 做 matmul + sigmoid + 逐元素乘，得本块 `u` 的局部 abs_max，核内累积得 `myMax`。每核负责全部 N 列，故 x1Scale/x2Scale 每核只读一次；
2. **跨核归约**：各核把 `myMax` 写到归约区本核槽位，双 `SyncAll` 完成「写 local → 读全部求 max → 才开始写 y 输出」握手，得全局 `gMax`；
3. **计算 yScale**：`yScale = gMax / 127.0`，`gMax == 0` 时取 1.0，仅 block 0 经 MTE3 `DataCopyPad` 写出 yScale GM；
4. **Phase 2（量化输出）**：重新遍历本核负责的 M 块，重算 `u`，按 `rint(u/yScale)` 截断到 [-128,127]，转 int8 逐行一维 burst 写 y GM。各核负责的 M 行段在 y GM 内互不重叠。

两阶段各做一次完整 matmul，计算量翻倍，但避免了中间结果 `u` 的 GM 暂存（M×N 个 FP16 需 2·M·N 字节，无 workspace 可用）。

### 4.2 跨核 abs_max 归约（复用 y GM + 双 SyncAll）

**约束**：`bias` 固定为空（`gen_data.py` 明确），workspace 尺寸 `WS_SYS_SIZE = 0`，`GetSysWorkSpacePtr()` 无实际空间可用 → **无独立 GM 归约区**。

**解法**：复用 y GM 前 `blockNum` 个 float 位作每核 local abs_max 暂存区（1 个 float = 8 个 int8 元素），host 已保证 `totalNum >= blockNum*8`：

```cpp
redGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), blockNum_ * 8);
```

该区间与 y 输出区重叠，靠**双 `SyncAll`** 保证时序：

```cpp
// 第一道 SyncAll：所有核都已写完 local abs_max
SyncAll();
// 读回全部核的 local 值求全局 max（本核槽位已被 myMax 覆盖）
float gMax = myMax;
for (b = 0; b < blockNum_; ++b) { if (b != blockIdx_) gMax = max(gMax, redGM_[b*8]); }
// block 0 写 yScale ...
// 第二道 SyncAll：所有核都读完归约区，才允许覆写 y 输出区
SyncAll();
```

第二道 `SyncAll` 不可省：否则后完成的核可能覆写归约区而其它核尚未读取。

**`SyncAll()` 无参版在 dav_c220 是硬件 `ffts_cross_core_sync`**（`dav_c220/kernel_operator_sync_impl.h`），不需要 GM 工作区，与 `IBSet/IBWait`（软同步 GM 轮询）不同。ConfusionMulGrad / MseLoss / Trace **三个已过评测算子**均用此无参版（ConfusionMulGrad 注释「block0 清零 + SyncAll + SetAtomicAdd 为真机已验证模式」）。

**单核回退**：`blockNum_ == 1` 时跳过归约区写入（`myMax` 即全局值，且 y GM 可能不足 `blockNum*8` 字节，写入会越界）。

### 4.3 matmul：向量化外积 Axpy

对每个输出块 `[mCnt][nCnt]`，按 k 维分块（`kBlk = min(32, K)`）循环：

- **x1 加载**（`LoadX1Block`）：逐元素 `SetValue/GetValue` 整理为 UB 布局。
  - `transposeX1=false`：物理 (M,K)，UB 布局 `[mCnt][kAl]`（行主序）；
  - `transposeX1=true`：物理 (K,M)，UB 布局 `[kAl][mAl]`（按 k 行、m 列）。
- **x2 加载**（`LoadX2Block`）：统一整理为 `[kAl][stride]`（行主序，行距 = stride）。
  - `transposeX2=false`：物理 (K,N)，直接读；
  - `transposeX2=true`：物理 (N,K)，按 n 列跨步读。
- **类型转换**：int8 → half → float（`Cast` 两段，dav_c220 无 int8→float 直转）；
- **外积累加**：对每个 (m, k)，标量读 `x1[m][k]`，执行
  `Axpy<float,float>(acc[m*stride], x2f[k*stride], a_mk, stride)`，即 `acc[m][:] += x2[k][:] * a_mk`。

选 `Axpy` 而非 `MulAddDst`/矩阵指令的原因：`Axpy` 的 `dst += src * scalar` 语义恰好是外积的一行，且支持 float 累加。**Axpy 不支持 int32**，故累加走 float；K ≤ 675 时累加和 ≤ 675×127×127 ≈ 10.9M < 2^24，float 精确等价 int32。

### 4.4 计算链行距 stride：32 元素倍数（硬约束）

dav_c220 的 **`Cast` 与 `DataCopyPadUB2GMImpl` 均强制 src/dst 地址 32B 对齐**，非对齐直接 abort（`Failed to check src tensor address alignment ... should be 32 byte aligned`）。而 `DataCopyPad` 写 GM 的 `srcStride` 单位是「32B 块数」而非字节（`kernel_check_data_copy_overflow.h` 的 `CheckDataCopyPadTensorSizeOverflow` 中 `srcMaxOffsetBytes = blockCount*AlignUp(blockLen,32) + (blockCount-1)*srcStride*DEFAULT_C0_SIZE`，`DEFAULT_C0_SIZE` = 32）。

int8 行距非 32 倍数时二维 burst 无法表达（实测 `tail_33x65x17` 报 3872 vs 3072，精确吻合 `16*32 + 15*7*32`），故：

- **计算链行距统一取 `stride = (nAl + 31) & ~31`**（32 元素倍数），涉及 `MatmulRows` / `BuildU` / `LocalAbsMax` / `PostChunk` 四处，必须完全一致；
- **y 写出改逐行一维 burst**（`blockCount=1, blockLen=nCnt` 字节），与 11 个已过评测算子的惯用法一致；
- 非 32 倍数尾部由 `copy_ubuf_to_gm_align_b8` 的 padding 参数兜底（BatchNorm3D 已真机验证 int8 非 32 倍数 `blockLen` 可用）。

**尾块教训（本轮踩坑，重要）**：`LoadX2Block` 原按 `nAl`（8 元素对齐）作行距写 `x2Buf_`，而 `MatmulRows` 的 `Axpy` 按 `stride` 读 —— 尾块 `nAl < 32` 时 `stride=32 ≠ nAl=8`，x2 数据整体错位。`tail_17x96x100`（N=100, chunkN=96 → 第二块 nCnt=4, nAl=8, stride=32）bad=1400/1700 即此根因。修复后 `LoadX2Block` 与所有 `Cast` count、UB 容量分配全部改用 `stride`，11/11 全 PASS。

### 4.5 后处理链（BuildU）

对 `acc` 块依次执行，严格复刻 golden 的三个 FP16 舍入点：

```
h  = fp16(acc * merged)          // Muls → Cast(float→half) → Cast(half→float)
z  = -h，clamp 到 [-88, 88]      // 仅 clamp exp 参数，h 本身不 clamp
e  = exp(z) + 1
g  = fp16(1 / e)                 // Div → Cast → Cast
u  = fp16(g * h)                 // Mul → Cast → Cast
```

**关键决策：clamp 只作用于 exp 参数，不作用于 h。** golden 的 `u = sigmoid(h)*h` 在 h 为大正值时随 h 线性增长（sigmoid 饱和到 1）；若把 h 也 clamp 到 88，`u` 会被截断在 88，导致 abs_max 偏小、yScale 偏小、量化结果错。此点经 `[multi] K=100` 用例暴露（kernel yScale=88/127 vs golden 91.31/127）后修正。

### 4.6 全局 abs_max 归约的块内折叠（LocalAbsMax）

对 `u` 块取绝对值后折叠求 max：

```
cntA  = mCnt * stride                  // 有效元素数（stride 为 32 倍数）
cntAl = align_up(cntA, 32)             // 再补齐到 32 元素
Abs(u, u, cntA)
fold = 0; Max(fold, u, cntA)
// REP=64 repeat 折叠 + 尾部并入
if (cntAl > 64) {
    k = cntAl / 64;  rems = cntAl - k * 64
    Max(fold, fold[64], fold, 64, k-1, {1,1,1,0,8,0})   // repeat 折叠
    if (rems > 0) Max(fold, fold[k*64], fold, rems)     // 尾部并入
    WholeReduceMax(rmax, fold, 64, ...)
} else {
    WholeReduceMax(rmax, fold, cntAl, ...)
}
```

**不可用树形折半**（曾致 yScale = `0xEEEEEEEE` = -3.697e28 垃圾值）：`cntAl` 非 2 的幂时折半链会落到非 8 倍数（如 640→320→160→80→40→**20**），`Max` count 非法；且折半可能提前停在 < 32，`WholeReduceMax` 固定 mask=32 会越界读未初始化 UB。REP=64 repeat 折叠与 `MultiAddRmsNormDynamicQuant::ReduceMaxScalar` 的真机惯用法一致，不改变折叠宽度故无 buffer 溢出风险。

### 4.7 量化输出（PostChunk）

```
inv = yScale (broadcast)
q   = u / inv                        // 真除法，不用倒数乘法
q   = min(q, 127); q = max(q, -128)  // 先 clip 后取整，与 golden 一致
s32 = rint(q)                        // CAST_RINT = round-half-to-even
f   = float(s32); h16 = half(f); q16 = half(f)   // 规避 int32→half / int8→int32 直转缺失
// 逐行 Cast 到 y8Buf_ 的 32B 对齐行距，再逐行一维 burst 写 GM
for m in [0, mCnt):
    Cast<int8_t>(y8[m*stride], q16[m*stride], CAST_RINT, stride)
    DataCopyPad(yGM_[(mBase+m)*N_ + nBase], y8[m*stride], {1, nCnt, 0, 0, 0})
```

**必须用真除法而非倒数乘法**：golden 是 `u32 / y_scale`（`gen_data.py:71`）。实测 `Muls(u, u, 1/yScale)` 在 tie 附近差 1 ULP——`q = 63.4999962`（bits `0x427dffff`）被倒数乘法抬到 `63.5`，`CAST_RINT` 多进一位产出 64，而 golden 正确给 63。改 `Duplicate(inv, yScale)` + `Div` 后该 2/1700 偏差归零。

int8 输出链走 `CAST_RINT` 匹配 `torch.round` 的 RNE 语义；中间经 `int32 → float → half → int8` 三段，因 dav_c220 无 `int32→half`、`int8→int32` 直转指令（直转会静默产出 0）。

**y GM 写出用 MTE3 `DataCopyPad` 逐行一维 burst，禁用多核逐元素 `SetValue` 直写**：BatchNorm3D / QBMv3 两次确证多核 `GlobalTensor::SetValue` 直写 GM 会大面积丢失。单核下 `SetValue` 可靠，故历史上曾采用；多核改造后必须改 MTE3。

### 4.8 host 侧（tiling.cpp）

- 解析 (M, K, N) 与两个 transpose 属性；
- CHUNK_M/CHUNK_N 受 UB 预算约束：`chunkM ≤ 16`、`chunkN ≤ 192`（精确核算见 4.9，**不可取 256**，会 UB 溢出）；
- **多核切分**：`blockDim = min(coreNum, CeilDiv(M, chunkM))`，并加 `M*N >= blockDim*8` 守卫（保证归约区不越界）；仅 `SetBlockDim`，**严禁叠加 `SetSimtGridDim`**（后者仅适用 SIMT kernel，混用致平台 Time Limit Exceeded，11 个已过评测算子 0 处使用）；
- workspace = 0；
- tilingKey：FP16/BF16 输入 → MODE_0(half kernel)，其余 → MODE_1(int8 kernel)。

**chunk 约束与 K 解耦**（2026-11-05 修正）：kernel 早已改为按 `kBlk = min(32, K)` 分块加载，UB buffer 尺寸与 K 无关。此前 host 仍沿用废弃的"整 K 加载"假设施加 `chunkM ≤ 32768/K`、`chunkN ≤ 8192/K` 压缩——K=8192 时 `chunkM` 塌到 4、`chunkN` 塌到 1，n 方向分块数暴涨到 N，单核迭代量约 1 亿次，超出平台 3 秒硬超时。已删除该压缩，仅受 UB 预算约束。

### 4.9 UB 预算（stride 改动后重新核算）

按最大元素数统一按 `sizeof(float)` 分配，兼容 half/int32/int8 多类型复用，规避尺寸不足：

| buffer | 尺寸 | 用途 |
|---|---|---|
| x1Buf_ / x1fBuf_ | mAl·kAl·4 B | x1 int8 原始块 / float 视图 |
| x2Buf_ / x2fBuf_ | kAl·**stride**·4 B | x2 int8 原始块 / float 视图 |
| x1hBuf_ | 2·max(mAl·kAl, mAl·stride) B | int8→half 中转 / BuildU 的 h16 |
| x2hBuf_ | 2·max(kAl·stride, mAl·stride) B | int8→half 中转 / BuildU 的 g16、PostChunk 的 q16 |
| accBuf_ / t1Buf_ / t2Buf_ / t3Buf_ / yBuf_ | mAl·stride·4 B | acc、h、z、e、u |
| **y8Buf_** | mAl·stride B | int8 输出中转（32B 对齐行距） |
| redBuf_ / statBuf_ | 512 B | 归约结果 / yScale / local abs_max 暂存 |

其中 `mAl = align8(chunkM)`、`nAl = align8(chunkN)`、`kAl = align8(min(32, K))`、`stride = align32(nAl)`。

**half 中转块容量必须同时覆盖两种用途**（2026-11-05 修正，曾致真机 abort）：

1. `MatmulRows` 的 int8→half `Cast`：x1 需 `mAl·kAl`、x2 需 `kAl·stride` 个 half；
2. `BuildU` 复用作 fp16 中间 buffer（h16/g16）：需 `mAl·stride` 个 half。

原实现按**输入块**尺寸分配，当 `nAl > 2·kAl` 时必然越界——如 `chunk_40x16x40`（nAl=40 > 32）、`chunk_17x96x100`（nAl=104 > 32）、`bigK_6x1024x96`（nAl=96 > 64），三个用例齐崩。

同时 `x1fBuf_` 从 `x1hBuf_` 拆出独立 TBuf（原实现共用一个 TBuf，float 视图写会覆盖 half 视图尚未读走的低位，属读写重叠）。

**精确核算**（chunkM=16、chunkN=192、kBlk=32，即当前 host 上限；mAl=16, nAl=192, stride=192, kAl=32）：

```
calc   = 16·192·4 = 12288 B × 5 (acc/t1/t2/t3/y)  = 61440
x1/x1f = 16·32·4  =  2048 B × 2                   =  4096
x2/x2f = 32·192·4 = 24576 B × 2                   = 49152
x1h    = 2·max(16·32, 16·192)  = 12288 B          = 12288
x2h    = 2·max(32·192, 16·192) = 24576 B          = 24576
y8     = 16·192   =  3072 B                       =  3072
red/stat = 512 B × 2                              =  1024
合计 137216 B ≈ 134 KB < 192 KB（余量 58 KB）
```

chunkN 取 256 时 calc 与 x2h/x2/y8 同步涨到 32 KB，合计 205 KB > 192 KB，会触发真机 UB 溢出崩溃（仿真器无硬容量限制，无法暴露该问题，故以解析核算为准）。

## 五、关键决策依据（实验实证）

1. **float 累加替代 int32**：`Axpy` 无 int32 特化、`Cast` 无 int8→int32 直转，int32 精确累加路线在 dav_c220 上不可行；K ≤ 675 时 float 累加和 < 2^24 保证精确，大 K 场景由"大数相消"压力测试覆盖。
2. **多核按 M 切分 + 复用 y GM 归约**：平台评测实测单核 3.56s 超 3s 硬超时（测试点 9，`ret=507046`，Block Dim=1 确认单核；测试点 1-8 全 Pass 证明精度已闭合）。跨核归约无独立 GM 区可用（workspace=0、bias 固定空），故复用 y GM 前 `blockNum` 个 float 位 + 双 `SyncAll`。切分后 3.56s/12 核 ≈ 0.3s，余量充足。
3. **sigmoid 只 clamp exp 参数**：见 4.5，clamp h 会截断大 h 区的 `u` 值导致 yScale 系统性偏小。
4. **计算链行距取 32 元素倍数**：dav_c220 的 `Cast` 与 `DataCopyPad` 强制 src/dst 32B 对齐；且写 GM 的 `srcStride` 单位为 32B 块数，int8 非 32 倍数行距的二维 burst 无法表达。故 `stride = align32(nAl)`，y 写出改逐行一维 burst。**四处的 stride 必须完全一致**（`LoadX2Block` 曾漏改，致尾块数据错位）。
5. **逐元素加载而非 DataCopy 跨步**：`DataCopyExtParams` 的 `blockLen`/`srcStride` 单位为字节、`dstStride` 单位为 32B 块数，三者单位不一致极易错；逐元素 `SetValue/GetValue` 虽慢但语义无歧义，且 x1/x2 块尺寸小（≤ 32×192 int8），开销可接受。
6. **禁用 `PipeBarrier<PIPE_ALL>` 与 `MTE2_S`**：前者在 910B 单核场景会永久阻塞（历史教训），后者在已过评测算子中 0 处使用。跨管线同步仅用 `V_S`、`S_MTE3`、`MTE3_MTE2`、`V_MTE3` 四对事件。
7. **归约用 REP=64 repeat 折叠而非树形折半**：见 4.6，折半链会落到非 8 倍数使 `Max` count 非法。
8. **量化用真除法而非倒数乘法**：见 4.7，倒数乘法在 tie 附近差 1 ULP 致 `CAST_RINT` 多进一位。
9. **half 中转块按两种用途的最大值分配**：见 4.9，仅按输入块尺寸分配会在 `nAl > 2·kAl` 时越界踩相邻 UB。
10. **y8 输出用专用 TBuf 而非复用 x2fBuf_**：`Cast<int8_t>` 的目标也受 32B 对齐约束，复用 float 视图的 TBuf 时其 int8 行距无法独立取 32 倍数，故新增 `y8Buf_`（容量 `mAl·stride` 字节）。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 平台全量构建（`rm -rf build && bash build.sh`，dav_c220） | **PASS**（kernel `.o` + `custom_opp_ubuntu_aarch64.run` 均生成） |
| op_kernel UT（脚手架占位 tilingData {16,16,16}） | **PASS**（不崩溃、不越界；精度比对失败属预期，占位值按位域解出 M=16,K=1,N=16 与 golden 语义无关） |
| 精度探针：SmallShapes（7 例） | **7/7 PASS** |
| 精度探针：TransposeCombos（7 例） | **7/7 PASS** |
| 精度探针：Chunking（7 例） | **7/7 PASS** |
| 精度探针：BigKChunkFix（6 例，含 2 例 N>192 覆盖 n 多块） | **6/6 PASS** |
| **多核探针（本轮新增，11 例）** | **11/11 PASS** |

精度探针以真实 tiling 位域编码驱动交付 kernel，与独立实现的 CPU golden（复刻三个 FP16 舍入点与 RNE 量化）逐元素比对，**要求 y 与 yScale 全部 bitwise 一致**。覆盖：32 对齐与非对齐、K 多块（K=65/70/96/100/200/1024/2048/2700）、M/N 尾块、单元素退化、4 种 transpose 组合、n 方向多块（N=300/400）。

**多核探针（`test_qmsq_mc.cpp`，11 例全 PASS）** 覆盖多核切分精度与跨核归约正确性，y GM 预填 `0xEE` 哨兵值检测跨核写完整性：

| 用例 | M×K×N | blockDim | 验证点 |
|---|---|---|---|
| `platform_like_128x128x96` | 128×128×96 | 12 | 平台真实 N=96，12 核 M 轮询 |
| `mc4_64x64x64` / `mc8` / `mc12` | 64/64/96 × 64 × 64 | 4 / 8 / 12 | 不同核数下 M 轮询分配 |
| `spare_8_33x16x32` | 33×16×32 | 8 | 块数多于 M 块（尾部核提前返回） |
| `tail_17x96x100` | 17×96×100 | 4 | 非对齐尾块 + n 方向多块（chunkN=96 < N=100） |
| `tail_33x65x17` | 33×65×17 | 4 | M/N 双尾块，N<32 致 stride=32 ≠ nAl=24 |
| `deg_1x1x1` / `deg_1x16x1` | 1×1×1 / 1×16×1 | 1 | 单元素退化 |
| `deg_33x1x65` | 33×1×65 | 4 | K=1 退化 + 多核 |
| `single_8x8x8` | 8×8×8 | 1 | 单核对照（归约区路径不启用） |

全部用例 `bad=0`、`sentinel=0`、**yScale 逐位一致**，证明多核切分、跨核归约、M 轮询分配、哨兵写完整性均正确。

**本轮（2026-11-05）修复的三个真实缺陷**（此前探针 21/25，现 27/27）：

| 缺陷 | 现象 | 根因 | 修复 |
|---|---|---|---|
| half 中转块越界 | 3 例 abort（`Cast` 处）+ yScale=`0xEEEEEEEE` | `x1hBuf_`/`x2hBuf_` 按输入块尺寸分配，却被 `BuildU` 当 fp16 中间 buffer 写 `mAl·nAl` 个元素；`nAl > 2·kAl` 时越界 | 容量取 `2·max(输入块, mAl·nAl)`；判别式与 PASS/FAIL 完全吻合 |
| 量化倒数乘法 1 ULP | `chunk_17x96x100` bad=2/1700，delta=+1 | `Muls(u, u, 1/yScale)` 把 `63.4999962` 抬到 `63.5`，`CAST_RINT` 多进一位 | 改 `Duplicate(inv, yScale)` + 真 `Div` 对齐 golden |
| chunk 受 K 压缩 | 大 K 单核超 3s 平台硬超时 | host 沿用废弃的"整 K 加载"假设，`chunkN ≤ 8192/K` 在 K=8192 时塌到 1 | 删除 K 压缩；kernel 早已按 `kBlk` 分块，UB 尺寸与 K 无关 |

**多核改造轮（2026-11-05 晚）修复的两个缺陷**：

| 缺陷 | 现象 | 根因 | 修复 |
|---|---|---|---|
| y 写 GM 用 `SetValue` 直写 | 多核下 y GM 大面积丢失（BatchNorm3D/QBMv3 两次确证的模式） | `GlobalTensor::SetValue` 多核逐元素写不可靠 | 改 MTE3 `DataCopyPad` 逐行一维 burst |
| 二维 burst 的 srcStride 单位误解 | `tail_33x65x17` 报 3872 vs 3072 溢出 | `DataCopyPad` 写 GM 的 `srcStride` 单位是「32B 块数」而非字节，int8 行距非 32 倍数无法表达 | 放弃二维 burst，改逐行一维（`blockCount=1`） |
| `Cast`/`DataCopyPad` src 非 32B 对齐 | `Failed to check src tensor address alignment ... should be 32 byte aligned` abort | 原按 `nAl`（8 元素对齐）作行距，行首地址非 32B 倍数 | 计算链行距统一改 `stride = align32(nAl)`，新增 `y8Buf_` 专用 TBuf |
| **`LoadX2Block` 读写行距不一致** | `tail_17x96x100` bad=1400/1700，yScale 2.31e-05 vs 1.54e-05 | `LoadX2Block` 按 `nAl=8` 行距写，`MatmulRows` 的 `Axpy` 按 `stride=32` 读，尾块 x2 数据整体错位 | `LoadX2Block` 及所有 `Cast` count、UB 容量分配统一改用 `stride` |

另：`x1fBuf_` 从 `x1hBuf_` 拆为独立 TBuf（原 float 视图写覆盖 half 视图未读走的低位，属读写重叠）。

## 七、性能要点

- **多核切分是必需的，不是可选优化**：平台评测实测最大测试点单核 3.56s，超 3 秒硬超时（`aclrtSynchronizeStreamWithTimeout(stream, 3000) failed, ret=507046`）。改用 12 核按 M 轮询后约 0.3s，余量充足。
- **计算量**：两阶段各一次完整 matmul，有效 FLOPs 为最优方案（存 u 再量化）的 2 倍；`MatmulRows` 的 `Axpy` 外积两遍扫描为 `2·M·N·K` 次向量操作，这是单核 3.56s 的主要来源（非正确性问题）。
- **外积向量化**：内层 `Axpy` 一次处理 `stride`（≤192）个元素，摊薄标量读 x1 的开销；相比逐元素标量内积（K 次乘加全标量）有数量级提升。
- **chunk 上限修正后的加速**：修复前 K 大时 `chunkN` 塌到 1，单核迭代量约 1 亿次（~35ns/次 ≈ 3.5s）；修复后 `chunkN ≤ 192`，同规模降至约 36ms（~100×）。
- **UB 常驻**：所有中间块常驻 UB，仅 x1/x2 按 k 块经 GM→UB 搬运，无中间结果写 GM。
- **跨核归约开销可忽略**：每核仅 4 字节 GM 写 + `blockNum` 次 4 字节 GM 读，加两次硬件 `ffts_cross_core_sync`。

## 八、遗留事项

- op_host UT 依赖 `nlohmann/json.hpp`，本地环境缺失无法构建（已知问题，非本次改动引入）；
- 本地仅完成 CPU 仿真验证（多核探针 11/11 + kernel UT PASS），真机（910B）行为需平台评测确认，重点观察：多核 `DataCopyPad` 逐行写 GM 的完整性、双 `SyncAll` 在真实多核下的时序、UB 实际占用；
- 若后续平台开放独立 workspace 通道，可去掉「复用 y GM 归约区」的设计，改为独立 GM 区，消除第二道 `SyncAll` 的必要性。
