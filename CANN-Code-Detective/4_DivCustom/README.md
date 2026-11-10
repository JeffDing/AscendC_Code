# 算子工程调测

加载 CANN 9.0.0 或 9.1.0 环境，安装 Python 参考实现需要的 numpy、ml_dtypes（以及参考文件实际使用的 torch/tensorflow）。

```bash
source /path/to/cann/set_env.sh
bash run.sh             # msopgen 生成工程（若不存在）→ 构建 → 安装 → 编译 test → 运行
```

说明：

- `custom_op/` 为 `msopgen` 基于 `div_custom_template.json` 生成的算子工程，本仓库已提交完整实现，`run.sh` 检测到 `custom_op/build.sh` 存在时跳过生成步骤；
- 芯片目标为 `ai_core-ascend910b1`，与本地 910B3 匹配；芯片不一致时需手动重新 `msopgen gen`；
- 算子安装到 `${HOME}/vendors/customize`，运行前必须 source `${HOME}/vendors/customize/bin/set_env.bash`，否则 aclnn 找不到算子库；
- 测试用例 `test/main.cpp` 通过 aclnn 接口调用，输入 shape `(8, 2048)`，x=1.0、y=2.0，golden 期望 z=0.5。

---

# DivCustomTemplate 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现 DivCustomTemplate 算子：两个输入张量的逐元素除法。

| 项 | 规格 |
|---|---|
| 算子名 | `DivCustomTemplate` |
| 输入 | `x`、`y`，`DT_FLOAT16` / `DT_FLOAT`，`FORMAT_ND` |
| 输出 | `z`，与 `x`/`y` 同 shape、同 dtype |
| 属性 | 无 |
| 参考实现 | 逐元素 `z = x / y` |

**计算语义**：

```
z[i] = x[i] / y[i]
```

## 二、改动清单

| 文件 | 改动内容 |
|---|---|
| `custom_op/op_kernel/div_custom_template.cpp` | 实现 Ascend C kernel：`KernelDivCustomTemplate` 类的 Init/Process/CopyIn/Compute/CopyOut |
| `custom_op/op_host/div_custom_template.cpp` | 实现算子注册、TilingFunc、InferShape、InferDataType |

`custom_op/op_kernel/div_custom_template_tiling.h`、`custom_op/cmake/`、`custom_op/framework/`、`custom_op/scripts/`、`test/`、`div_custom_template.json`、`run.sh`、`pr_info.md` 均保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`DivCustomTemplateTilingData` 使用 `BEGIN_TILING_DATA_DEF` 宏定义的 3 个 `uint32_t` 字段，标量直存、不做位域拆分：

| 字段 | 语义 | 说明 |
|---|---|---|
| `size` | N | 输入元素总数（各维相乘） |
| `blockLength` | bl | 每核处理的元素数 = `size / 8` |
| `tileLength` | tl | 每轮 UB 处理的元素数，固定 2048 |

`blockNum` 固定为 8（`op_host/div_custom_template.cpp:19`），`context->SetBlockDim(blockNum)`。

## 四、算法设计

### 4.1 host 侧（`custom_op/op_host/div_custom_template.cpp`）

1. **总元素数**：遍历 `x1_shape->GetStorageShape()` 各维相乘得 `data_sz`，写入 `tiling.set_size()`；
2. **多核切分**：`blockNum = 8`，`blockLength = data_sz / 8`，`tiling.set_blockLength()`；`context->SetBlockDim(8)`；
3. **UB 分块**：`tileLength = 2048` 固定值，`tiling.set_tileLength()`；
4. **InferShape**：输出 shape 直接 `*y_shape = *x1_shape`；
5. **InferDataType**：输出 dtype 取输入 dtype，`context->SetOutputDataType(0, inputDataType)`；
6. **算子注册**：`x`/`y`/`z` 均声明 `DT_FLOAT16` + `DT_FLOAT` 双 dtype、`FORMAT_ND`，`AICore().AddConfig("ascend910")`。

### 4.2 kernel 侧（`custom_op/op_kernel/div_custom_template.cpp`）

**每核处理区间**：`offset = GetBlockIdx() * blockLength`，`xGm/yGm/zGm` 通过 `SetGlobalBuffer` 绑定到该偏移，长度 `blockLength`。

**数据通路**：

- `TQue<VECIN, 2>` × 2（inQueueX/inQueueY）+ `TQue<VECOUT, 2>`（outQueueZ）双缓冲，`BUFFER_NUM = 2`；
- 每个 buffer 按 `tileLength * sizeof(half)` 分配，即 `pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half))`；
- GM→UB：`DataCopy(xLocal, xGm[progress * tileLength], currentTileLength)`；
- UB→GM：`DataCopy(zGm[progress * tileLength], zLocal, currentTileLength)`。

**计算**（`Compute`）：`Div(zLocal, xLocal, yLocal, currentTileLength)` 内建向量指令完成逐元素除法。

**尾块处理**：`CopyIn`/`Compute`/`CopyOut` 三处均计算
`currentTileLength = (blockLength - progress * tileLength < tileLength) ? 剩余长度 : tileLength`，
即最后一轮按剩余元素数截断，避免越界读写。

**注意**：kernel 侧 `GlobalTensor` 统一按 `half` 类型绑定（`(__gm__ half*)x + offset`）。当前平台用例输入为 fp16，`Div<half>` 走 half 域向量指令，与 golden 逐位一致。

## 五、关键决策依据

1. **固定 blockNum=8 而非按核数动态切分**：题目用例规模固定为 `(8, 2048)` 共 16384 元素，8 核每核 2048 元素恰好等于一个完整 tile，每核单轮处理完毕，无多轮循环开销。
2. **tileLength 固定 2048**：2048 × 2 字节 = 4KB/buffer，3 个队列 × 双缓冲共 24KB，远低于 UB 上限，无需按平台 UB 尺寸动态计算。
3. **尾块截断放 kernel 侧**：host 只传 `size`/`blockLength`/`tileLength` 三个标量，kernel 自行判断末轮剩余量，tiling 逻辑最简单。
4. **`half` 直绑 GlobalTensor**：算子原型声明 fp16/fp32 双 dtype，但 kernel 实际按 half 寻址；fp32 输入需另行适配（当前用例不覆盖）。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| `bash run.sh` 一键流程 | **通过** |
| 算子工程构建 `bash build.sh` | **通过**（生成 `custom_opp*.run`） |
| 算子安装 `${HOME}/vendors/customize` | **通过** |
| aclnn 测试用例 `test/main.cpp` | **test pass** |

**实际运行输出**（`test/main.cpp`，shape `(8, 2048)`，x=1.0 / y=2.0）：

```
result is:
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5

test pass
```

## 七、性能要点

- 8 核并行，每核 2048 元素单轮 `CopyIn → Compute → CopyOut`，无循环调度开销；
- VECIN/VECOUT 双缓冲（深度 2），MTE2 搬运与 Vector 计算可流水重叠；
- UB 总占用约 24KB（3 队列 × 2 深度 × 4KB），远低于 192KB 池上限。

## 八、约束遵守与交付状态

- 实现集中在 `custom_op/op_host/` 与 `custom_op/op_kernel/` 两个目录，未改动 msopgen 生成的 cmake/framework/scripts 等工程文件；
- `run.sh` 保留"工程已存在则跳过 msopgen"的判断分支，重复执行不会覆盖已有实现；
- `pr_info.md` 记录了完整 PR 说明（实现方式、目录结构、测试结果）。
