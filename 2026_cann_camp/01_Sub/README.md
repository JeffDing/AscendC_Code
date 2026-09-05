# 个人开发者

## 团队信息

- 提交者: JeffDing
- 身份: 企业员工
- 单位: 个人开发者

## 成员

- JeffDing (JeffDing): 提交者

## 算子: op_01_sub

本算子实现逐元素减法 `z = x - y`，采用 Ascend C 矢量编程范式。Host 侧 `run_kernel` 从 `TensorGroupInfo` 推导总元素数 `totalLength`，固定 `tileNum=8`、`blockDim=8`，按输入 dtype 实例化 `sub_custom&lt;float&gt;` 或 `sub_custom&lt;half&gt;` 并通过 `&lt;&lt;&lt;blockDim, nullptr, stream&gt;&gt;&gt;` 启动核函数。Device 侧 `KernelSub&lt;T&gt;` 采用「CopyIn → Compute → CopyOut」三段式流水：`Init` 按 `GetBlockNum()`/`GetBlockIdx()` 将 Global Memory 切分到各核，并将每核数据再按 `tileNum × BUFFER_NUM` 分块（Double Buffer，`BUFFER_NUM=2`）；`CopyIn` 用 `DataCopy` 把当前块从 GM 搬入 UB 的 `TQue&lt;VECIN&gt;` 队列，`Compute` 调用 `AscendC::Sub` 完成矢量减法并写入 `TQue&lt;VECOUT&gt;` 队列，`CopyOut` 把结果搬回 GM，循环 `tileNum*BUFFER_NUM` 次处理完全部数据。
