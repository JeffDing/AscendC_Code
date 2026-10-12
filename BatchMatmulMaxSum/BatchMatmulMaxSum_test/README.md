# BatchMatmulMaxSum 本地测试设施

对 `/workspace/cann/BatchMatmulMaxSum/kernel.asc`（Ascend C 算子 kernel）的本地验证工程：15 个官方 case 的正确性回归、单 case 稳定性连跑、数据/golden 重新生成。**独立于目标工程，不参与提交。**

## 目录结构

```
BatchMatmulMaxSum_test/
├── CMakeLists.txt        # 精简构建配置（只构建 bmm_multi）
├── main_multi.asc        # 多 case 驱动：./bmm_multi <caseId 1..15>
├── data_utils.h          # 文件 I/O 工具
├── kernel.asc            # 被测 kernel 副本（从目标工程拷贝，见下文"同步 kernel"）
├── gen_data15.py         # 重新生成 15 case 输入 + FP64 golden
├── check_one.py          # 单 case 判分（rtol=atol=1e-3，输出 max_rel）
├── report.sh             # 一键测试报告：正确性回归 + 性能实测 + 稳定性连跑
├── flaky.sh              # 单 case 稳定性连跑（每次独立进程）
├── input/case1..15/      # 各 case 的 x1.bin / x2.bin（65MB）
└── output/golden_case1..15/  # FP64 golden（y 为 fp32 标量）
```

## 环境要求

- CANN（ascend-toolkit）已 source：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- Python：`/usr/local/python3.12.13/bin/python3`（需 numpy；系统 python3 没有 numpy，勿用）
- 昇腾 NPU 设备（本设施在 910B2 / dav-2201 上验证）

## 编译

```bash
cd /workspace/cann/BatchMatmulMaxSum_test
mkdir -p build && cd build
cmake .. -DNPU_ARCH=dav-2201
make bmm_multi -j4        # 产物：build/bmm_multi
```

### `-DNPU_ARCH` 如何取值

`NPU_ARCH` 是 Ascend C 编译的芯片架构版本，CMake 会传给 `ccec --npu-arch=...`，**必须与运行机器的 NPU 匹配**，否则 kernel 无法运行或行为异常。不传该参数时默认 `dav-2201`（Atlas A2 训练系列，Ascend 910B；本设施在 910B2 上验证）。

获取自己环境取值的方法（任选其一）：

1. **查 NPU 型号**：`npu-smi info` 看 `Name` 列（如 910B2、310P、950PR），再对照 CANN 文档《Ascend C API 参考》中的“芯片架构版本（npu-arch）”表。
2. **查 CANN 自带的架构表**（本机 CANN 9.1.0 实测路径，其他版本路径类似）：
   ```bash
   grep -n "dav-" /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/ascendc_kernel_cmake/asc_modules/CMakeASCInformation.cmake
   ```
   其中 simulator / CPU 仿真模式支持的架构：`dav-2002`（Ascend310P1）、`dav-2201`（Ascend910B1）、`dav-3510`（Ascend950PR_9599）。
3. **问编译器**：`/usr/local/Ascend/ascend-toolkit/latest/bin/ccec --help`，查看 `--npu-arch` / `--npu-soc` 说明；传一个错误值（如 `--npu-arch=bad`）也会被拒绝并提示无效。

已知对应（以所用 CANN 版本的官方文档为准）：

| NPU 型号 | npu-arch |
|---|---|
| Ascend 910B（A2 训练系列，含 910B2） | dav-2201 |
| Ascend 310P1 | dav-2002 |
| Ascend 950PR | dav-3510 |

注意：换硬件环境编译时务必改这个参数——用默认值能编译通过，但上板运行会失败。

## 使用方法

### 1. 15 case 正确性回归

```bash
cd /workspace/cann/BatchMatmulMaxSum_test
bash report.sh 5 1         # 正确性回归(reps=5) + 稳定性各 1 次
```

正确性回归环节：每 case 以 reps 个独立进程各跑 1 次后判分（OJ 同构口径），stdout 输出 `case12: PASS (y[0]=60286.5703, max_rel=0.00e+00)`，末尾 `---- 15 passed, 0 failed`。

### 2. 单 case 稳定性连跑（查偶发错误）

```bash
bash flaky.sh 12 20        # case12 连跑 20 次独立进程，统计通过率
```

widen 引擎验收口径：关键 case（c9-c12）需 20/20。

### 3. 单 case 手动运行

```bash
cp input/case12/x1.bin input/x1.bin
cp input/case12/x2.bin input/x2.bin
./build/bmm_multi 12                       # 写 output/y.bin
REPS=5 ./build/bmm_multi 12                 # 同进程连跑 5 次，stderr 输出计时
/usr/local/python3.12.13/bin/python3 check_one.py 12   # 判分
```

### 4. 一键生成完整测试报告（markdown）

```bash
bash report.sh              # 默认：正确性回归(reps=5) + 性能实测 + c9/c10/c11/c15 各连跑 20 次
bash report.sh 3 10         # 自定义：回归 reps=3，稳定性各 10 次
```

跑完在当前目录生成 **`TEST_REPORT.md`**，包含三个章节：

- **正确性回归表**：每 case 的 y[0]、max_rel、launch/单次 total（单位 us）
- **性能实测表**：每 case 设备侧 event 计时 best-of-8（单位 us）、OJ TBest 目标（us）、达成率（<100% 即优于官方目标）
- **稳定性表**：关键 case（c9/c10/c11/c15）独立进程冷启动连跑通过率
- 附：关键修复说明（async 引擎 SyncAll 修复）

单独跑 `flaky.sh` 只会向报告的稳定性章节追加结果（report.sh 为全量重写）。

### 5. 重新生成数据与 golden（一般不用，input/ 已内置）

```bash
/usr/local/python3.12.13/bin/python3 gen_data15.py .
```

种子与官方 harness 一致（20260928 + caseId，uniform [-1,1]），golden 为 FP64 计算结果。

## 同步 kernel（重要）

`kernel.asc` 是本设施的**副本**。修改目标工程后必须重新拷贝再编译：

```bash
cp /workspace/cann/BatchMatmulMaxSum/kernel.asc /workspace/cann/BatchMatmulMaxSum_test/kernel.asc
cd /workspace/cann/BatchMatmulMaxSum_test/build && make bmm_multi -j4
```

## 注意事项

- **单 NPU 只能串行测**：不要同时跑多个 `bmm_multi` 进程，会互相干扰导致假失败。
- **判分读的是 output/y.bin**：若进程挂起（超时/崩溃），main 可能在写 y.bin 前就退出，此时 y.bin 是上一次的旧文件，会造成"假通过"。脚本已用 `timeout 70` + 删除旧 y.bin 规避；自行排查时务必检查进程退出码和 y.bin 时间戳。
- **驱动写 16KB y 窗口**：main_multi.asc 的 resultBytes/outBytes 为 16384，高于 OJ 的 4B，仅本地诊断用，不影响 kernel 正确性验证。
- 目标工程提交只接受 `kernel.asc` 一个文件的变更，本目录内容不提交。
