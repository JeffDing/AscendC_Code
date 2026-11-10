# 算子工程调测

加载 CANN 9.0.0 或 9.1.0 环境，安装 Python 参考实现需要的 numpy、ml_dtypes（以及参考文件实际使用的 torch/tensorflow）。

```bash
source /path/to/cann/set_env.sh
bash build.sh -j4       # 构建 RUN 包
DEVICE_ID=0 bash build.sh -j4 -e  # 构建并运行首个测试点、比较输出
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include GTEST_SOURCE_DIR=/path/to/googletest bash build.sh -j4 -u # 本地 UT
```

GoogleTest 使用本地源码（建议1.14.0），nlohmann/json 头文件建议3.11.3（include目录下应有nlohmann/json.hpp），不自动下载。CPU 调测还需要工具链的 tikicpulib。修改 op_host、op_kernel，以及存在时的 op_api。kernel为空时输出校验失败是正常现象。samples 只包含第一个 NPU 测试点。输入按题目规范生成，比较前检查文件大小。

opapi 工程的完整结果由外层 NPU example 校验；内层 CPU UT 的 fixture 需要按实际内外层转换关系自行提供，不能直接套外层 GT。默认 -u 明确跳过内层数值断言。准备并编辑内层 fixture 后可使用 `ENABLE_INNER_KERNEL_UT=ON bash build.sh -u` 启用；其结果需按内层语义自行断言。API UT 需要设备上下文，默认明确跳过；先构建 RUN，再使用 `RUN_OP_API_UT=1 DEVICE_ID=0 bash build.sh -u` 执行外层 workspace 接口测试，该测试不代替 -e 的数值校验。若修改 tiling 结构，也需要同步调整 kernel UT 的 tiling fixture。

---

# Abs 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现 Abs 算子：对输入逐元素取绝对值。

| 项 | 规格 |
|---|---|
| 算子名 / 包名 | `Abs` |
| 输入 | 单输入 `x`，`DT_FLOAT` / `DT_FLOAT16`，`FORMAT_ND`，任意维 shape |
| 输出 | 单输出 `y`，与 `x` 同 shape、同 dtype |
| 属性 | 无 |
| 参考实现 | 逐元素 `abs(x)` |

**计算语义**：

```
y = |x|
```

**平台 golden**（`tests/ut/op_kernel/abs_data/gen_data.py`）：float64 中间计算后一次性转回输入 dtype。

**tilingKey**（`schMode`）：`0` → fp16/bf16 kernel，`1` → fp32 kernel。

## 二、改动清单（仅限含 TODO 标记的文件，共 3 个）

| 文件 | 改动内容 |
|---|---|
| `op_kernel/abs.h` | 实现 kernel 全部逻辑（模板类 `Abs<T>` 的 Init/Process/CopyIn/Compute/CopyOut） |
| `op_host/abs_tiling.cpp` | 实现 tiling 计算（多核切分、ubFactor 固定 1024、tilingKey 选择） |
| `op_host/abs_infershape.cpp` | 输出 shape 与输入相同 |

**未修改任何其他文件**：`build.sh`、`CMakeLists.txt`、`op_host/abs_def.cpp`、`op_host/CMakeLists.txt`、`op_kernel/abs_tiling_data.h`（无 TODO）、`op_kernel/abs_tiling_key.h`、`op_kernel/abs.cpp`、`tests/`、`examples/` 全部保持原样。

## 三、TilingData 三字段的语义约定（结构未动）

`AbsTilingData` 保持原有 3 个 `int64_t` 字段，标量直存：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalNum` | N | 输入元素总数 |
| `blockFactor` | bf | 每核处理的元素数 |
| `ubFactor` | uf | 每轮 UB 处理的元素数 |

host 侧计算（`op_host/abs_tiling.cpp:80-106`）：

1. `totalNum` = 输入 shape 各维相乘（空 shape 兜底为 1）；
2. **多核切分**：`usedCore = min(coreNum, totalNum)`，`blockFactor = CeilDiv(totalNum, usedCore)`，再以 `blockFactor` 反算 `usedCore = CeilDiv(totalNum, blockFactor)`；空张量时收敛回 1 避免 `SetBlockDim(0)`；
3. **UB 分块**：`ubFactor = 1024` 固定值，再收敛到不超过 `blockFactor`，下限 1。

注意本算子 host 侧**不按 32B 对齐** `blockFactor`（与 Exp/Sqrt 不同），对齐依赖 kernel 侧 `DataCopyPad` 处理非对齐边界。

## 四、算法设计

### 4.1 infershape（`op_host/abs_infershape.cpp`）

模板生成结构：取 `input_shape = context->GetInputShape(0)`，单循环 `for (i = 0; i < 1; i++)` 内 `*output_shape = *in_shape`，空指针逐层返回 `GRAPH_FAILED`。

### 4.2 kernel 侧（`op_kernel/abs.h`）

**每核处理区间**（`Init`，`:52-78`）：

```cpp
totalNum_    = tilingData->totalNum;
blockLength_ = max(tilingData->blockFactor, 1);
ubLength_    = clamp(tilingData->ubFactor, 1, ABS_UB_FACTOR_CAP);  // CAP = 2048
inputGM_.SetGlobalBuffer(x, totalNum_);   // 注意：绑定全长而非本核段
outputGM_.SetGlobalBuffer(y, totalNum_);
blockIdx_ = GetBlockIdx();
blockNum_ = GetBlockNum();
offset_   = blockIdx_ * blockLength_;
```

关键点：`SetGlobalBuffer` 绑定**整个张量长度** `totalNum_`，本核区间通过 `offset_` 在 `Process` 中寻址，而非在 `Init` 里偏移。这种写法下 `blockNum_` 被取出但当前 `Process` 未使用（预留字段）。

**UB 容量钳制**（`Init`，`:64-67`）：`ABS_UB_FACTOR_CAP = 2048`，注释说明理由——双队列各 2 深度共 4 份缓冲，合计 `4 * cap * sizeof(T)` 字节须 < 192KB UB 池；`cap = 2048` 时 fp32 占 32KB / fp16 占 16KB，留有充足余量。本地 UT 的占位 `ubFactor` 可能远超该值，必须在此收敛。

**数据通路**（入方向 `CopyIn`，`:82-89`）：

```cpp
DataCopyExtParams  copyParams{1, currentNum * sizeof(T), 0, 0, 0};
DataCopyPadExtParams<T> padParams{true, 0, 0, 0};   // pad = true
DataCopyPad(in, inputGM_[offset_ + progress], copyParams, padParams);
```

出方向 `CopyOut`（`:105-111`）用 `DataCopyPad` 3 参形式（无 padParams 重载）。

**计算**（`Compute`，`:94-101`）：

```cpp
AscendC::Abs(out, in, static_cast<int32_t>(currentNum));
```

显式限定 `AscendC::` 命名空间，避免与注入的类名 `Abs<T>` 产生歧义（源码注释已说明）。

**循环**（`Process`，`:114-131`）：

```cpp
for (int64_t i = 0; i < blockLength_; i += ubLength_) {
    int64_t cur = ubLength_;
    int64_t offset = offset_ + i;
    if (offset >= total) break;              // 超出总长直接退出
    if (offset + cur > total) cur = total - offset;  // 尾块截断
    CopyIn(i, cur); Compute(cur); CopyOut(i, cur);
}
```

`i` 是本核内相对偏移，传给 `CopyIn/CopyOut` 后与 `offset_` 相加得全局偏移；`offset >= total` 的 break 覆盖尾核数据不足的场景。

## 五、关键决策依据

1. **`ubFactor` 固定 1024 而非按 ubSize 计算**：host 侧不查询平台 UB 尺寸，用固定值 + kernel 侧 `ABS_UB_FACTOR_CAP = 2048` 双重收敛。1024 元素 × 4 份 × 4 字节 = 16KB（fp32），任何平台都不会溢出，代价是大张量轮次略多（16384 元素需 16 轮）。
2. **`SetGlobalBuffer` 绑全长 + `offset_` 寻址**：与 Relu/Exp/Sqrt 的"Init 里偏移"写法不同。此写法下 `inputGM_[offset_ + progress]` 的索引是相对绑定起点的绝对偏移，语义更直观，且 `Process` 中的 `offset >= total` 判断可直接用全局坐标。
3. **入方向 `padParams{true, ...}`**：pad 置 true 使非对齐尾部按 0 填充写入 UB，避免读取 GM 越界；出方向不需要 pad（写多少是多少）。
4. **`AscendC::Abs` 显式命名空间**：类名 `Abs<T>` 与内建函数 `Abs` 同名，不限定命名空间时编译器可能解析到类模板，加 `AscendC::` 消歧义。
5. **显式保存 `blockNum_`**：当前 `Process` 未用，但取出该字段为后续改为"核数不足时多轮处理"（`for (blk = blockIdx; blk < totalBlocks; blk += blockNum)`）预留了条件，与 host 侧"核数不足时每核多轮"的注释呼应。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| 平台构建 `bash build.sh -j4` | **通过**（ascend910b） |
| CPU 仿真 UT（tikicpulib Ascend910B1）kernel UT | 通过 |
| CPU 仿真 UT sample verify | 通过 |

**UT 数据**（`tests/ut/op_kernel/abs_data/gen_data.py`）：输入 `np.ones((8, 2048))` 共 16384 元素，golden 为 float64 中间计算后转回输入 dtype；`compare_data.py` 逐元素比对。

**UT 占位值注意**：本地 kernel UT 硬编码的 `ubFactor` 可能远超 `ABS_UB_FACTOR_CAP`，此时 UT 数值仍应正确（因钳制逻辑生效），但**不得**因 UT 通过就删除钳制代码——它是防退化 tiling 越界的必要保护。

## 七、性能要点

- 单指令 `AscendC::Abs` 完成逐元素计算；
- VECIN/VECOUT 双缓冲（深度 2），MTE2 搬运与 Vector 计算流水重叠；
- `ubFactor` 固定 1024（fp32 下单轮 4KB × 4 份 = 16KB UB），大张量分多轮，调度开销高于 Exp/Sqrt 的大 ubFactor 方案；
- 多核按 `CeilDiv(totalNum, min(coreNum, totalNum))` 均分，核数不足时每核多轮。

## 八、约束遵守与交付状态

- 仅修改三个白名单文件（`op_kernel/abs.h`、`op_host/abs_tiling.cpp`、`op_host/abs_infershape.cpp`），未改动任何其他已有源码文件；
- `op_kernel/abs_tiling_data.h`、`op_kernel/abs_tiling_key.h`、`op_kernel/abs.cpp`、`op_host/abs_def.cpp` 均无 TODO，保持原样；
- 本工程 `tests/ut/` 下自带 `nlohmann/json.hpp`（其他算子工程无此文件），UT 构建不依赖外部 json 路径。
