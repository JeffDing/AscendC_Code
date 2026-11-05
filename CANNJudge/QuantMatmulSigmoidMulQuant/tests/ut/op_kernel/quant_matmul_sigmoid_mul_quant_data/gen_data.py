#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch

_INT8_LOW = -128
_INT8_HIGH = 127


def impl(x1, x2, x1Scale, x2Scale, bias=None, transposeX1=False, transposeX2=False):
    """QuantMatmulSigmoidMulQuant算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
        x1: INT8 二维 ND Tensor，transposeX1=False 时物理形状 (M, K)，True 时 (K, M)
        x2: INT8 二维 ND Tensor，transposeX2=False 时物理形状 (K, N)，True 时 (N, K)
        x1Scale/x2Scale: shape (1,) 的 FLOAT32 T-T 反量化标量
        bias: 必须为 None；本题不支持 bias
        transposeX1/transposeX2: bool，仅描述物理布局

    返回与输入同为 numpy 或同为 torch 的输出:
        y: (M, N) int8 量化结果；yScale: (1,) float32 全矩阵动态量化 scale。
    """
    return_numpy = not isinstance(x1, torch.Tensor)
    a = torch.as_tensor(x1)             # numpy / torch 输入统一转 torch
    b = torch.as_tensor(x2)
    s1 = torch.as_tensor(x1Scale)
    s2 = torch.as_tensor(x2Scale)

    # bias 固定为空（None 或零元素数组）
    if bias is not None:
        bias_t = torch.as_tensor(bias)
        if bias_t.numel() > 0:
            raise ValueError("bias is fixed to None in this topic")

    # 由物理存储布局建立逻辑矩阵视图；不生成额外的转置输入
    if bool(transposeX1):
        a = a.t()
    if bool(transposeX2):
        b = b.t()

    # 阶段一：INT8 矩阵乘，INT32 精确累加
    acc = torch.matmul(a.to(torch.int32), b.to(torch.int32))

    # 阶段二：FLOAT32 合并 T-T scale，结果落到 FP16（舍入点 1）
    merged = s1.float().reshape(-1)[0] * s2.float().reshape(-1)[0]
    h = (acc.float() * merged).to(torch.float16)

    # 阶段三：稳定 sigmoid（FLOAT32，负值分支避免 exp 大数溢出），落回 FP16（舍入点 2）
    z = h.float()
    pos = z >= 0
    sig = torch.empty_like(z)
    sig[pos] = 1.0 / (1.0 + torch.exp(-z[pos]))
    exp_z = torch.exp(z[~pos])
    sig[~pos] = exp_z / (1.0 + exp_z)
    g = sig.to(torch.float16)

    # 阶段四：FP16 逐元素相乘（FLOAT32 中转），结果落回 FP16（舍入点 3）
    u = (g.float() * h.float()).to(torch.float16)

    # 阶段五：全矩阵动态 per-tensor 量化（RNE 先舍入、后 clip）
    u32 = u.float()
    abs_max = u32.abs().amax()
    y_scale = abs_max / 127.0
    if float(y_scale) == 0.0:
        y_scale = torch.ones_like(y_scale)      # abs_max == 0 时约定取 1.0
    y = torch.clamp(torch.round(u32 / y_scale), _INT8_LOW, _INT8_HIGH).to(torch.int8)
    y_scale = y_scale.reshape(1)                # shape (1,)，严禁按行/按列输出多个 scale

    if return_numpy:
        return y.numpy(), y_scale.numpy()
    return y, y_scale


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "int8"
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
    input_x1 = np.ones((1, 16)).astype(d_type_dict["int8"])
    input_x2 = np.ones((16, 1)).astype(d_type_dict["int8"])
    input_x1Scale = np.ones((1)).astype(d_type_dict["float32"])
    input_x2Scale = np.ones((1)).astype(d_type_dict["float32"])
    input_bias = None  # 可选输入，未提供
    attr_transposeX1 = False
    attr_transposeX2 = False
    
    # 计算 golden 数据
    golden = impl(input_x1, input_x2, input_x1Scale, input_x2Scale, input_bias, attr_transposeX1, attr_transposeX2)
    
    # 保存数据到文件
    input_x1.astype(d_type_dict["int8"]).tofile(f"{d_type}_input_quant_matmul_sigmoid_mul_quant_x1.bin")
    input_x2.astype(d_type_dict["int8"]).tofile(f"{d_type}_input_quant_matmul_sigmoid_mul_quant_x2.bin")
    input_x1Scale.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_quant_matmul_sigmoid_mul_quant_x1Scale.bin")
    input_x2Scale.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_quant_matmul_sigmoid_mul_quant_x2Scale.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            _out_dtypes = ["int8", "float32"]
            for _gi, _g in enumerate(golden):
                _dt = _out_dtypes[_gi] if _gi < len(_out_dtypes) else _out_dtypes[-1]
                _g.astype(d_type_dict[_dt]).tofile(f"{_dt}_golden_quant_matmul_sigmoid_mul_quant_{_gi}.bin")
        else:
            golden.astype(d_type_dict["int8"]).tofile("int8_golden_quant_matmul_sigmoid_mul_quant_0.bin")
    
    print(f"生成完成: dtype={d_type}")
