#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch


def impl(gradOutput, normOut, sumOut):
    """MhcSinkhornGrad算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
        gradOutput: (B,S,n,n) 或 (T,n,n) numpy 数组或 torch.Tensor，上游梯度，float32
        normOut:    (2*numIters, T, n, n) 前向保存的归一化中间量，float32，T = B*S
        sumOut:     (2*numIters, T, n) 前向保存的归约和（按 3.1 协议生成），
                    sumOut[0] 为保留位不参与计算，float32

    返回与输入同为 numpy 或同为 torch 的输出:
        gradInput: 与 gradOutput 同 shape，float32

    迭代反传:
        列归一反向（下标 2i+1）：dot = Σ_dim=-2(grad ⊙ normOut[2i+1])，
                                 grad ← (grad - dot) / sumOut[2i+1]（按列广播）
        行归一反向（下标 2i）  ：dot = Σ_dim=-1(grad ⊙ normOut[2i])，
                                 grad ← (grad - dot) / sumOut[2i]（按行广播）
    末轮:
        列归一反向（下标 1）  ：同列归一反向；
        softmax 反向（下标 0）：gradInput ← (grad - Σ_dim=-1(grad ⊙ normOut[0])) ⊙ normOut[0]
    """
    return_numpy = not isinstance(gradOutput, torch.Tensor)
    go = torch.as_tensor(gradOutput)    # numpy / torch 输入统一转 torch
    norm_t = torch.as_tensor(normOut)
    sums_t = torch.as_tensor(sumOut)

    # 全部计算采用 FP32 流水，不得降精度
    shape = go.shape
    orig_dtype = go.dtype
    T = int(np.prod(shape[:-2]))
    n = int(shape[-1])
    num_iters = int(norm_t.shape[0]) // 2

    g = go.float().reshape(T, n, n).clone()
    norm = norm_t.float().reshape(2 * num_iters, T, n, n)
    sums = sums_t.float().reshape(2 * num_iters, T, n)

    # 前 numIters-1 轮：先列（2i+1）后行（2i），i 从高到低
    for i in range(num_iters - 1, 0, -1):
        dot = (g * norm[2 * i + 1]).sum(dim=-2, keepdim=True)   # 按列投影
        g = (g - dot) / sums[2 * i + 1][:, None, :]
        dot = (g * norm[2 * i]).sum(dim=-1, keepdim=True)       # 按行投影
        g = (g - dot) / sums[2 * i][:, :, None]

    # 末轮：列归一反向（下标 1）
    dot = (g * norm[1]).sum(dim=-2, keepdim=True)
    g = (g - dot) / sums[1][:, None, :]
    # softmax 反向（下标 0）：乘 normOut[0]，无除法
    dot = (g * norm[0]).sum(dim=-1, keepdim=True)
    grad_input = (g - dot) * norm[0]

    out = grad_input.reshape(shape).to(orig_dtype)

    if return_numpy:
        return out.numpy()
    return out


def _scalar_reference(grad, norm, sums, num_iters):
    """纯 python 列表标量参考实现（独立于 torch 向量化路径，用于自检）。"""
    n = len(grad)
    g = [[float(v) for v in row] for row in grad]

    def col_dot(gm, nm):
        return [[sum(gm[i][j] * nm[i][j] for i in range(n)) for j in range(n)] for _ in range(n)]

    def row_dot(gm, nm):
        return [[sum(gm[i][j] * nm[i][j] for j in range(n))] * n for i in range(n)]

    for i in range(num_iters - 1, 0, -1):
        dot = col_dot(g, norm[2 * i + 1])
        g = [[(g[r][c] - dot[r][c]) / sums[2 * i + 1][c] for c in range(n)] for r in range(n)]
        dot = row_dot(g, norm[2 * i])
        g = [[(g[r][c] - dot[r][c]) / sums[2 * i][r] for c in range(n)] for r in range(n)]

    dot = col_dot(g, norm[1])
    g = [[(g[r][c] - dot[r][c]) / sums[1][c] for c in range(n)] for r in range(n)]
    dot = row_dot(g, norm[0])
    return [[(g[r][c] - dot[r][c]) * norm[0][r][c] for c in range(n)] for r in range(n)]


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float32"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]
    
    # 生成输入数据
    input_gradOutput = np.ones((1, 4, 4)).astype(d_type_dict["float32"])
    input_normOut = np.ones((2, 1, 4, 4)).astype(d_type_dict["float32"])
    input_sumOut = np.ones((2, 1, 4)).astype(d_type_dict["float32"])

    
    # 计算 golden 数据
    golden = impl(input_gradOutput, input_normOut, input_sumOut)
    
    # 保存数据到文件
    input_gradOutput.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_mhc_sinkhorn_grad_gradOutput.bin")
    input_normOut.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_mhc_sinkhorn_grad_normOut.bin")
    input_sumOut.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_mhc_sinkhorn_grad_sumOut.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_mhc_sinkhorn_grad.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_mhc_sinkhorn_grad.bin")
    
    print(f"生成完成: dtype={d_type}")
