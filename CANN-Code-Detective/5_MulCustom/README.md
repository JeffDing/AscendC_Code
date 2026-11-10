# 算子工程调测

加载 CANN 环境，安装 Python 参考实现需要的 numpy、ml_dtypes（以及参考文件实际使用的 torch/tensorflow）。

```bash
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
mkdir -p build && cd build/
cmake .. && make && ./mul_test
```

或直接执行一键脚本：

```bash
bash run.sh
```

说明：

- 本工程是 **Ascend C kernel 直调**形态：`mul_custom.asc` 单文件同时包含 kernel（`__global__ __aicore__ void mul_custom`）与 host 侧封装（`kernel_mul`），host 通过 `<<<blockDim, nullptr, stream>>>` 语法直接发射 kernel，不经过 aclnn/算子注册；
- 编译目标 `--npu-arch=dav-2201`（对应 AscendC 直调工程），非标准 `ascend910b` 算子工程；
- 无 op_host/op_kernel 目录拆分，无 TilingData 注册宏，无 samples/tests/examples 脚手架。

---

# MulCustom 算子实现报告

## 一、算子规格

在昇腾 NPU（Ascend C）上实现逐元素乘法算子：两个输入张量相乘。

| 项 | 规格 |
|---|---|
| 算子语义 | `z[i] = x[i] * y[i]` |
| 输入 | `x`、`y`，float32，长度 `totalLength` |
| 输出 | `z`，float32，长度 `totalLength` |
| 属性 | 无 |
| 参考实现 | 逐元素 float 乘法 |

**计算语义**：

```
z[i] = x[i] * y[i]
```

**用例数据**（`main`，`mul_custom.asc:169-178`）：`totalLength = 8 * 2048 = 16384`，`x` 全 1.2f、`y` 全 2.3f，golden 全 `2.76f`。

## 二、改动清单

| 文件 | 改动内容 |
|---|---|
| `mul_custom.asc` | 实现 kernel 全部逻辑：`KernelMul` 类成员/Init/Process/CopyIn/Compute/CopyOut、`mul_custom` 入口、`kernel_mul` host 封装、`VerifyResult` 与 `main` |

`CMakeLists.txt`、`run.sh` 保持原样。

## 三、TilingData 两字段的语义约定（结构未动）

`MulCustomTilingData` 为 2 个 `uint32_t` 标量直存：

| 字段 | 语义 | 说明 |
|---|---|---|
| `totalLength` | N | 输入元素总数 |
| `tileNum` | tn | 每核切分的 tile 份数，固定 8 |

kernel 入口签名 `mul_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, MulCustomTilingData tiling)`，tiling 按值传递（非指针）。

## 四、算法设计

### 4.1 kernel 侧（`mul_custom.asc`）

**每核处理区间**：`blockLength = totalLength / GetBlockNum()`，三路 `SetGlobalBuffer` 分别偏移 `blockLength * GetBlockIdx()`，即 8 核均分，每核 2048 元素。

**UB 切分**：`tileLength = blockLength / tileNum / BUFFER_NUM = 2048 / 8 / 2 = 128`，即每核 8 个 tile × 双缓冲 = 16 轮循环。

**数据通路**：

- `TQue<VECIN, 2> inQueueX, inQueueY` + `TQue<VECOUT, 2> outQueueZ`，`BUFFER_NUM = 2`；
- 每 buffer 按 `tileLength * sizeof(float)` = 512 字节分配；
- GM→UB：`DataCopy(xLocal, xGm[progress * tileLength], tileLength)`；
- UB→GM：`DataCopy(zGm[progress * tileLength], zLocal, tileLength)`。

**计算**（`Compute`）：`Mul(zLocal, xLocal, yLocal, tileLength)` 内建向量指令。

**循环**（`Process`）：`loopCount = tileNum * BUFFER_NUM = 16`，顺序执行 `CopyIn → Compute → CopyOut`。

### 4.2 host 侧（`kernel_mul`）

标准 ACL 直调流程：`aclInit` → `aclrtSetDevice` → `aclrtCreateStream` → `aclrtMallocHost` ×3 / `aclrtMalloc` ×3 → `aclrtMemcpy` H2H + H2D → `mul_custom<<<blockDim, nullptr, stream>>>(...)` → `aclrtSynchronizeStream` → `aclrtMemcpy` D2H → 释放 → `aclFinalize`。`blockDim = 8`。

### 4.3 验证（`VerifyResult` / `main`）

`std::equal` 逐元素精确比较 output 与 golden，打印前 20 个值辅助定位，返回值作为进程退出码。

## 五、关键决策依据

1. **kernel 直调而非标准算子工程**：学习型样例，无 op_host 注册/InferShape/Tiling 分发链路，单文件即可编译运行，验证 Ascend C 基础接口用法。
2. **tileNum=8 双缓冲 → 16 轮循环**：`blockLength / tileNum / BUFFER_NUM` 的切分公式使 tile 数 × 缓冲深度恰好循环 `tileNum * BUFFER_NUM` 次，每轮处理 `tileLength` 元素；该公式要求 `blockLength` 能被 `tileNum * BUFFER_NUM` 整除（本用例 2048/16=128 恰整除）。
3. **tiling 按值传递**：`mul_custom(..., MulCustomTilingData tiling)` 第四参直接传结构体值而非 `GM_ADDR`，这是直调形态的简化写法，省去 `GET_TILING_DATA` 宏。
4. **全常量输入数据**：x=1.2、y=2.3 为编译期常量，golden 由 `valueX * valueY` 直接算得，便于精确比较定位 kernel 逻辑错误。

## 六、验证结果

| 验证项 | 结果 |
|---|---|
| CMake 配置 + 编译 `cmake .. && make` | **通过**（`--npu-arch=dav-2201`） |
| 运行 `./mul_test` | **`[Success] Case accuracy is verification passed.`** |

## 七、性能要点

- 8 核并行，每核 2048 元素分 16 轮处理，单轮仅 128 元素；
- VECIN/VECOUT 双缓冲使 MTE2 搬运与 Vector 计算可流水重叠；
- 每 buffer 512 字节，3 队列 × 2 深度共 3KB UB，占用极低（tile 粒度偏小，调度开销占比高于生产算子）。

## 八、约束遵守与交付状态

- 仅实现 `mul_custom.asc` 单文件内的 TODO 骨架，未改动 `CMakeLists.txt` 与 `run.sh`；
- kernel 入口、host 封装、验证主函数三部分齐备，`bash run.sh` 可一键完成环境加载、构建、运行。
