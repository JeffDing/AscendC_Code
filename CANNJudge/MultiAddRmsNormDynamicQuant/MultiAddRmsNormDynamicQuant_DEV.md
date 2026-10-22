# MultiAddRmsNormDynamicQuant 算子优化——最终交付文档

> 交付日期：2026-10-22
> 修改文件（仅三个白名单文件）：
> - `op_host/multi_add_rms_norm_dynamic_quant_infershape.cpp` — **未改动**
> - `op_host/multi_add_rms_norm_dynamic_quant_tiling.cpp` — **未改动**
> - `op_kernel/multi_add_rms_norm_dynamic_quant.h` — **全部修改在此**

## 1. 成果

| 项 | 修复前 | 修复后 |
|---|---|---|
| 本地 910B 真机复现（B=1024,H=8192，8.4M 元素） | y2 有 3~4 个 int8 翻转 | **y1/y2/scale1/scale2/x/y 全部 bitwise 0 差异** |
| 10 组 shape 终验（含 H=5、质数 33/2053/8191、攒批 B=8192、H=16384） | — | **10/10 全 PASS** |
| 编译 | — | `build.sh` 0 error |
| kernel 耗时（B=1024,H=8192） | ~1.18ms | ~2.41ms（2.05x，正确舍入指令的固有代价） |

## 2. 根因与修复（五处，全部"sim 全过、真机错"）

评测为 **int8 bitwise 无容差**比对（浮输出为 1e-4 容差）。kernel 必须**全链逐位复刻 torch CPU**。
本地 NPU=Ascend 910B2 与评测平台一致，通过 `build.sh` → op 包安装 → `examples/test_aclnn_*.cpp`
（aclnn 真机）→ torch 金标逐位比对，闭环复现并定位：

### 2.1 ✅ sum 归约结构 ≠ torch CPU `cascade_sum`
金标 `mean = sum_out(...).div_(numel)`；sum 走 `vectorized_inner_sum`
（`SumKernel.cpp`，aarch64 实测 **ilp=4, tail_ilp=1, W=4, num_levels=4, level_power=4**）：
- level0：`acc[0..16)` 按 **16 元素连续块顺序累加**（`sizeIlp=H/16` 块）——即逐块 `Add(count=16)`
- 级联提升：每 16 块（=256 元素）`acc1+=acc0` 并清零；每 256 块 `acc2+=acc1`；每 4096 块 `acc3+=acc2`
- 尾部向量（`[sizeIlp*4, vecSize)` 逐 4-lane）+ 标量尾 + 合并顺序严格对齐
**实现**：`TorchCascadeAdd / TorchCascadeFinal`（新增 `bufAcc_` 128 float）。
真机 dump sumSq 对 `(x*x).sum(-1)`：**0/1024 行差**（H=16/256/8192）。

### 2.2 ✅ 910B 无 IEEE FMA：`FusedMulAdd`(vmadd) 为两步舍入
探针实证 `s0=1.7180096f, x=2.9515572f`：`vmadd` 的 `s0*s0-x` = **-2.384e-7**
（= fl(s0*s0)-x，乘法先舍入），精确值 -2.043e-7。→ 精确残差改用 **Dekker TwoProd
（12-bit split：`c=fl(x*4097); xh=c-fl(c-x); xl=x-xh`，hi+lo 乘积精确）**。

### 2.3 ✅ rstd 的 sqrt 与 1/x（vsqrt/vdiv 单指令近似 ~0.6ulp）
rstd 段内联 Dekker Newton：
- `s = s0 + (x - TwoProd(s0)) / (2·s0)` → **sqrt 与 torch 0/1024 差**
- `r = q0 + (1 - TwoProd(s,q0)) / s` → **rstd 与 torch 0/1024 差**
- **mean = sumSq/h 也必须精化**（H 非 2 幂时 vdiv 差 1ulp——曾致 H=2053 的 scale 差 27 行；
  内联 Dekker 除法后归零）

### 2.4 ✅ Pass3 的 q÷scale 与 scale=max/127（vdiv 近似）
`TorchDiv`（通用）/`TorchDivC`（除数常量版：`bh/bl` 行级 Dekker 预计算 + 广播段，
块内省 4 条）：`q0=IeeeDiv; e=a−TwoProd(b,q0); q=q0+e/b` → 0ulp。
calc 段布局（bufCalc_ 6144 float）：`[0,1K)q [1K,2K)q/s [2K,3328)TorchDivC w
[3328,4416)RNE w [4416..)div/bh/bl 广播（路间覆写）`。

### 2.5 ✅ `Cast<float,int32> CAST_RINT` 真机为 round-half-**away**（sim 是 RNE！）
`torch.round` 是 RNE → 精确 .5 处翻 1 LSB（复现中的 885/946/955 三个翻转全因此）。
**手动 RNE（9 条向量指令/块，纯算术 + `And` 奇偶）**：
```
r  = CAST_RINT(x)            // away
d  = x - float(r)
sgn= CAST_RINT(d)            // ∈{-1,0,1}，仅 |d|==0.5 时非 0（符号=修正方向）
odd= r & 1                   // And(c1I)
adj= sgn * odd               // sgn*|sgn|≡sgn → tie 因子冗余（数学简化）
r' = TRUNC(float(r) + adj)   // RNE
```
分块 128；`SetDeqScale` 后保留 1 个 barrier（SPR 生效时序）。

## 3. 性能（2.05x 的构成与已做优化）

组件分解（B=1024,H=8192）：基线 1.18ms + cascade/Dekker 0.33ms + TorchDivC 0.48ms + RNE 0.40ms。
已做：去掉全部非必要 `PipeBarrier`（V pipe FIFO 保序，实测 barrier 非瓶颈）、`And` 替代 6 条
算术求奇偶、`sgn*|sgn|≡sgn` 省一条 Mul、RNE 13→9 条、TorchDivC 常量除数版、除法分块 256。
未做（收益/风险比低）：RNE 块 256 的 calc 段布局超界（bl 段 6272>6144）、cascade 的
Add(16)×H/16 为 torch 结构刚需不可合并。

## 4. 已知遗留（非本次引入，原版遗留）

1. **bf16 路径系统性错误（~96% 输出错，含 x）**：用原版 kernel 复测同样全错 → 原版遗留
   （疑似 bf16 输入 Cast/搬运层面）。评测错误占比 0.00% 级说明 **15 个评测 case 均为 fp16**；
   若评测含 bf16 case 需另行排查（治具已支持 bf16 参数：`test_aclnn <data> B H bf16`）。
2. 治具说明：`examples/test_aclnn_*.cpp` 被改造为真机验证治具（读 bin 输入、参数化 B/H/dtype、
   kernel 计时、回写 y1/y2/scale/x/y），**非算子交付物**（评测不涉及 examples）。

## 5. 可复用结论

- **int8 bitwise 评测 = kernel 全链逐位对齐 torch CPU**；`tikicpulib` 仿真（CPU IEEE）无法暴露
  910B 的：`vsqrt/vdiv/vrec` 近似、`vmadd` 两步舍入、`CAST_RINT` away 舍入——必须真机闭环。
- 910B 精确浮点工具箱：**Dekker TwoProd（无 FMA 时的精确乘积）** → Newton 精化 sqrt/div；
  **`sgn=RINT(d)` 兼做 tie 检测与方向**（sgn*|sgn|≡sgn）；**`And` 求 int 奇偶**。
- torch CPU 归约结构以 `SumKernel.cpp` 的 `cascade_sum` 为准（不要假设顺序/折半树）。
- LocalTensor 段偏移必须 **8 float（32B）对齐**，否则真机 vector core exception 507035
  （`The UB address accessed by the VEC instruction is not aligned`）——曾因 `w[1]` 4B 步进崩溃。
- 大坑记录：kernel 挂掉时 `Sync failed 507035` 且**输出文件为旧残留**——验证前务必
  `rm -f out_*.bin` 并检查 test 程序 exit code，否则会得出"改动无效"的误判（本次曾因此
  误判 And/Compare/Select 失效，浪费多轮）。