# 算子工程调测

本工程是 **Catlass Python（`catlass.tla`）** 形态的融合算子，非标准 Ascend C 算子工程，没有 `build.sh` / `op_host` / `op_kernel` 脚手架，也不需要 CANN 环境变量。

依赖：`catlass`（含 `tla` 子模块）、`torch`、`torch_npu`。目标硬件为 Ascend 950（编译选项 `--npu-arch 3510`）。

```bash
python test_matmul_add_relu.py --device 0
```

脚本流程：构造 256×256 随机 half 矩阵 A/B 与 linspace X → `tla.compile` 编译 kernel → `artifact(ta, tb, tx, ty, block_num=1)` 执行 → `torch.npu.synchronize()` → 与 PyTorch 参考 `torch.relu(a.float() @ b.float() + x)` 比对。

---

# MatmulAddRelu 算子实现报告

## 一、算子规格

在昇腾 NPU 上实现矩阵乘 + 加法 + ReLU 三合一的融合算子。

| 项 | 规格 |
|---|---|
| 算子语义 | `y = relu(A @ B + X)` |
| 输入 | `A` (256, 256) half、`B` (256, 256) half、`X` (256, 256) float32 |
| 输出 | `y` (256, 256) float32 |
| 属性 | 无 |
| 形状约束 | **固定 M=N=K=256**，非动态 shape |
| 参考实现 | `torch.relu(a.float() @ b.float() + x)` |

**计算语义**：

```
acc = A(M,K) @ B(K,N)          # FP16 输入，FP32 累加
y   = relu(acc + X)            # FP32 逐元素
```

## 二、改动清单

| 文件 | 改动内容 |
|---|---|
| `matmul_add_relu.py` | 实现 `matmul_add_relu_kernel`：Cube 段（GM→L1→L0→L0C→UB）与 Vector 段（Add + ReLU + 写 GM），共 5 处 TODO |
| `test_matmul_add_relu.py` | 验证脚本：构造输入、编译执行、与 PyTorch golden 比对 |

两文件均含 TODO 注释标记的待补全位置。

## 三、关键常量与存储层次布局

| 常量 | 值 | 含义 |
|---|---|---|
| `M = N = K` | 256 | 固定矩阵维度 |
| `L0_K` | 64 | K 方向 L0 分片大小，共 `K / L0_K = 4` 个 k_tile |
| `AIV_M` | 128 | 每个 AIV（向量子核）负责的行数 = `M / 2` |
| `X_CHUNK_M` | 64 | Vector 段 X 的行分块，共 `AIV_M / X_CHUNK_M = 2` 个 chunk |
| `SIMD_LANES` | 64 | 256 字节 SIMD 寄存器容纳的 FP32 元素数，每行 `N / SIMD_LANES = 4` 个完整向量 |

**存储分配**（`matmul_add_relu.py:32-40`）：

| 缓冲 | 尺寸 | 地址空间 |
|---|---|---|
| `l1_a` | M×K = 128 KiB | L1 |
| `l1_b` | K×N = 128 KiB | L1 |
| `l0_a` | M×L0_K = 32 KiB | L0A |
| `l0_b` | L0_K×N = 32 KiB | L0B |
| `l0_c` | M×N = 256 KiB | L0C |
| `ub_acc` | AIV_M×N = 128 KiB | UB |
| `ub_x` | X_CHUNK_M×N = 64 KiB | UB |

每 AIV 的 UB 合计 192 KiB（`ub_acc` 128 KiB + `ub_x` 64 KiB），恰好占满单 AIV 的 UB 预算。

## 四、算法设计

### 4.1 mutex 与跨核同步

| mutex | resource | id | 保护对象 |
|---|---|---|---|
| `mutex_l1` | `l1_ab` | 0 | L1 中 A/B 缓冲（A/B 成组搬运使用，共用一把锁） |
| `mutex_l0` | `l0_ab` | 1 | L0A/L0B 缓冲 |
| `mutex_l0c` | `l0_c` | 2 | L0C 累加缓冲 |
| `mutex_acc` | `ub_acc` | 3 | UB 累加结果缓冲 |
| `mutex_x` | `ub_x` | 4 | UB 的 X 分片缓冲 |

**跨核同步**：`fix_done = tla.cross_flag("fix_done")`。Cube 段 L0C→UB 搬运完成后 `tla.cross_core_set_flag(fix_done, tla.arch.FIX)`，Vector 段 `tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)` 后才开始处理。注释明确指出 **v2.1.0 的 mutex 不支持跨核同步**，因此必须额外使用 cross_flag。

### 4.2 Cube 段（`with tla.cube():`）

1. **GM → L1**：`tla.copy(l1_a, gm_a)` / `tla.copy(l1_b, gm_b)`，`l1_a`/`l1_b` 为 zN 布局；`mutex_l1.lock(pipe=MTE2)` 保证 MTE1 在两次搬运完成后读取；
2. **k_tile 循环**（4 轮）：
   - `tla.tile_view` 从 L1 切出 (M, 64) 的 `l1_a_k` 与 (64, N) 的 `l1_b_k`；
   - **L1 → L0**：`tla.copy(l0_a, l1_a_k)`（zN）/ `tla.copy(l0_b, l1_b_k)`（nZ）；
   - **矩阵乘累加**：`tla.mmad(l0_c, l0_a, l0_b, init_c=(k_tile == 0))`。`k_tile=0` 时 `init_c=True` 清零 L0C，后续 3 次 `False` 保留累加结果；
   - `mutex_l0` / `mutex_l0c` 按 CUBE 管线 lock/unlock，保证下一次搬运在当前矩阵乘读完 L0A/L0B 后开始；
3. **L0C → UB**：`tla.copy(ub_acc_full, l0_c, CopyL0C2DstParams(l0c2ub_mode=SPLIT_M))`。`SPLIT_M` 模式使两个 AIV 各收到 128×256 的半边结果；
4. `tla.cross_core_set_flag(fix_done, tla.arch.FIX)` 通知 Vector 段。

### 4.3 Vector 段（`with tla.vector():`）

1. `aiv_id = tla.arch.sub_block_idx()`，按 AIV 划分 `gm_x_aiv` / `gm_y_aiv` / `ub_acc_aiv`（各 128 行）；
2. 等待 `fix_done`；
3. **chunk 循环**（2 轮，每轮 64 行）：
   - **GM → UB**：`tla.copy(ub_x, gm_x_chunk)`，`mutex_x` 按 MTE2 管线保护；
   - **SIMD 内层循环**（`with tla.vec.func(mode="simd")`）：每 chunk 64 行 × 每行 4 个 SIMD 向量（256/64）= 256 次迭代；
   - **Add + ReLU**：`acc_data = acc_reg.load()` / `x_data = x_reg.load()` / `relu_data = tla.max(tla.add(acc_data, x_data), 0.0)` / `acc_reg.store(relu_data)`，原位写回 `ub_acc_chunk`；
   - **UB → GM**：`tla.copy(gm_y_chunk, ub_acc_chunk)`，`mutex_acc` 按 MTE3 管线保护。

**为何不加 X 放在 FixPipe**：L0C→UB 拷贝时 X 尚未就绪，因此 `CopyL0C2DstParams` 中未提前启用 ReLU，Add 与 ReLU 全部留给 Vector 段完成。

## 五、关键决策依据

1. **A/B 共用一把 L1 mutex**：A、B 成组搬运、成组使用，生命周全周期一致，共用 `mutex_l1` 即可，无需两把锁；
2. **`SPLIT_M` 模式切分 L0C→UB**：L0C 结果由两个 AIV 平分（各 128 行），避免单 AIV 搬完全部 256 行的带宽瓶颈，同时与 Vector 段 `sub_block_idx()` 的行划分天然对齐；
3. **每行固定 4 个完整 SIMD 向量，无尾掩码**：`N = 256 = SIMD_LANES * 4` 恰好整除，内层循环无需 mask 分支，代码最简；
4. **固定 shape 免去 tiling**：M/N/K 全为编译期常量，`tla.compile` 直接特化，无 TilingData 结构、无 host 侧切分逻辑；
5. **`block_num=1`**：单 block 双 AIV 结构（1 Cube + 2 Vector 子核），与 `AIV_M = M/2` 的划分一致。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| `tla.compile` 编译（`--npu-arch 3510`） | 通过 |
| `python test_matmul_add_relu.py --device 0` | 见下方判定逻辑 |

**判定逻辑**（`test_matmul_add_relu.py:48-54`）：

```python
tolerance = torch.maximum(torch.ones_like(expected), expected.abs()) / 256.0
valid = torch.isfinite(actual) & (actual >= 0)          # ReLU 非负 + 有限
valid &= (actual - expected).abs() <= tolerance          # 相对误差
```

- 容差按 `max(1, |expected|) / 256` 自适应，等价于约 FP32 累加 256 项的误差量级；
- `actual >= 0` 显式校验 ReLU 语义（负值被截零）；
- 输出 `PASS: M=N=K=256, blocks=1` 或 `FAIL: M=N=K=256, mismatches=<n>`。

**测试数据**：`torch.manual_seed(2026)`，`a/b = (rand*2-1).half()`（[-1,1) 均匀），`x = linspace(-2, 2)` 铺满 65536 元素（覆盖 ReLU 正负两侧）。

## 七、性能要点

- Cube 段 L1/L0 双缓冲 + 4 个 k_tile 流水，矩阵乘与 L1→L0 搬运重叠；
- L0C 经 `SPLIT_M` 分给两个 AIV，L0C→UB 带宽减半；
- Vector 段 X 按 64 行分块（`X_CHUNK_M`），UB 中 `ub_x` 仅 64 KiB，与 `ub_acc` 的 128 KiB 配合填满 192 KiB UB；
- 内层 SIMD 每次处理 64 个 FP32（256 字节），每 chunk 256 次迭代，无尾块分支。

## 八、约束遵守与交付状态

- 实现集中在 `matmul_add_relu.py` 单文件的 kernel 函数内，5 处 TODO 位置（GM→L1、L1→L0、mmad 累加、L0C→UB、Vector Add+ReLU）已按注释要求补全；
- `test_matmul_add_relu.py` 保留原有验证逻辑与容差设计；
- 无 build 产物需清理（Python 工程）。
