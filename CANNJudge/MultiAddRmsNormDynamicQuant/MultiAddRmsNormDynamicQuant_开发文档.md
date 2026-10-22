# MultiAddRmsNormDynamicQuant 算子优化开发文档

> 更新日期：2026-10-21
> 涉及文件（仅白名单三文件）：
> - `op_host/multi_add_rms_norm_dynamic_quant_infershape.cpp`（未改）
> - `op_host/multi_add_rms_norm_dynamic_quant_tiling.cpp`（未改）
> - `op_kernel/multi_add_rms_norm_dynamic_quant.h`（本轮修改）

## 1. 问题

真机评测 case1-11 Pass、**case12-15 Wrong Answer（输出错误占比显示 0.00%）**。
评测标准（`tests/ut/op_kernel/.../compare_data.py`）：浮点输出用 MERE/MARE 容差（宽松），
**int8 输出 bitwise 逐位比对、无任何 LSB 容差**。因此只需极少元素在 `rint` 半整数边界
翻转 1 LSB，即判 WA（被 8M~33M 元素稀释显示为 0.00%）。

## 2. 方法：搭建 910B 真机复现环境

本地 NPU 为 **Ascend 910B2**（`npu-smi`），与评测平台一致。据此搭建闭环：

1. `bash build.sh -j8` 编译算子并生成 op 包；
2. `bash build/custom_opp_openEuler_aarch64.run --quiet --force` 安装；
3. `examples/test_aclnn_*.cpp`（aclnn 调用，**非 tikicpulib 仿真**）跑真机；
4. 用 `/usr/local/python3.12.13/bin/python3` + torch(CPU) 生成金标并逐位比对。

复现用例 `B=1024,H=8192`（8.4M 元素）：**y2 出现 3~4 个 int8 翻转 / 8.4M**，与 case12-15 现象一致。

## 3. 根因定位（两处，均为「sim 全过、真机错」）

### 3.1 根因一：sum 的归约结构与 torch CPU 不一致（已修复）

`gen_data.py` 金标：`var = (x32*x32).mean(dim=-1)` → torch CPU。
经查 PyTorch 源码（`SumKernel.cpp` / `ReduceOps.cpp`，jsDelivr 拉取）：

- CPU 的 `mean` = `sum_out(...).div_(numel)`（**真除法**）；
- `sum` 走 `cascade_sum` → `vectorized_inner_sum`（aarch64 实测分支
  **`ilp_factor=4`、`tail_ilp_factor=1`、vec width=4**，`num_levels=4`），即：

  | 阶段 | 结构 |
  |---|---|
  | level0 | `acc[0][0..16)` 按 **16 元素连续块顺序累加**，共 `sizeIlp=H/16` 块 |
  | cascade 提升 | 每 16 个 i 步（=256 元素）`acc1+=acc0` 并清零；`i%4096==0` 时 `acc2+=acc1`；`i%65536==0` 时 `acc3+=acc2` |
  | 尾部/合并 | 尾部向量、标量尾、`acc0+=(acc1,acc2,acc3)`、`ps0[l]=a[l]+a[4+l]+a[8+l]+a[12+l]`、逐 lane 汇总，**全部严格按 torch 顺序** |

用 numpy 精确复刻该结构，对 H=4…8192 **全部 bitwise 匹配 torch**（含复现数据 rstd 0/1024 差异）。

**实现**（kernel .h）：`TorchCascadeAdd`（逐 16 元素块累加 + 级联提升，向量 `Add(count=16)`
+ `Duplicate` 清零）+ `TorchCascadeFinal`（合并/尾部在标量域）。新增 `bufAcc_`（128×4B）。
Pass1 的 `sumSq` 由原折半树/`ReduceSumScalar` 改为该级联结构。

**验证**：真机 dump `sumSq` 对比 `(x*x).sum(-1)`，H=16/256/8192 **0/1024 差异**；rstd 差异行数由
原 226/1024 降至 69/1024（±1ulp）。

### 3.2 根因二：910B 的 `vsqrt`/`vdiv` 非 IEEE 正确舍入（未完全修复）

`rstd = 1.0/sqrt(mean+eps)`。torch CPU 用 **IEEE 正确舍入**的 sqrt 与真除法；
而 AscendC `Sqrt<float>`/`Div<float>` 在 910B 直接编译为 `vsqrt`/`vdiv` 单指令（`dav_c220` 头文件确认），
**910B 无 `SqrtConfig`/`DivConfig` 高精度档**（该 config 仅 950/`dav_3510` 支持）。

真机测得：`vsqrt` 结果与 IEEE sqrt **约 0.6 ulp**（63/1024 行差 1ulp），
`mean = sumSq/h` 在 H 为 2 的幂时精确（0/1024），即除法在 2 幂下无差。

**尝试过的精化**（均用 910B `FusedMulAdd`= `vmadd`，探针实测语义为 **`dst = src0*dst_old + src1`**）：
一次 Newton `s = s0 + (x-s0²)/(2s0)`（FMA 精确残差）与二次 Newton，均仍残留 63 行 1ulp 差异——
`hi = fl(s0+corr)` 在 `|corr|≤0.5ulp` 时被 RNE 吸收，无法纠正 s0 偏 0.5ulp 的行；
正确舍入 sqrt 需要 float-float（Dekker TwoSum）+ 尾数奇偶判定的最后修正，尚未实现。

## 4. 当前状态

| 项 | 结果 |
|---|---|
| `sumSq` 归约 | **100% bitwise 对齐 torch**（所有 H） |
| `rstd` | 残留约 6.9% 行 ±1ulp（**vsqrt 近似**导致） |
| 复现 y1/y2 翻转 | 0 / **3**（原 0 / 4） |
| case12-15 真机 | 预计仍 WA（翻转率 ~3.6e-7，35M 元素 case 必碰撞） |
| 编译 | `build.sh` 0 error |

## 5. 下一步（消除最后 1ulp）

在 kernel 内实现 **IEEE 正确舍入的 sqrt/div**（使 rstd 与 torch 逐位一致）：

1. **float-float（Dekker）正确舍入 sqrt**：`s0=vsqrt(x)` → FMA 残差 → Newton 得 `(hi,lo)` →
   依 `lo` 符号与 `ulp(hi)`（由指数位提取）做 tie-to-even 的 ±1ulp 修正；
2. 或 **正确舍入除法**：`q0=a/b` → FMA 精确残差 → `q1=q0+r/b` → 同样的最后修正；
3. 之后 `sumSq`（已对齐）+ 对齐后的 `rstd` → 全链 bitwise → case12-15 应转 Pass。

## 6. 关键结论（可复用）

- 该评测为 **int8 bitwise**，必须让 kernel 全链与 **torch CPU** 逐位一致，
  不能用仿真（`tikicpulib`，CPU IEEE）代替真机——910B 的 SFU 近似指令（`vsqrt`/`vdiv`/`vrec`/`vrsqrt`）
  与 CPU IEEE 存在 ≤1ulp 差异，是「sim 全过、真机 WA」的典型来源。
- `FusedMulAdd` 在 910B 语义为 `dst = src0*dst_old + src1`（`vmadd`），非 `src0*src1+dst`。
- torch CPU 的归约结构（cascade_sum）与常见假设（纯顺序 / 简单折半树）**均不同**，
  必须按源码（或逐级 bitwise 逆向）精确复刻。