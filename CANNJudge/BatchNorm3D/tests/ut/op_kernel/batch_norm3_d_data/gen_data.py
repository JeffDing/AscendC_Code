#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(x, scale, offset, epsilon=1e-5):
    """BatchNorm3D算子golden实现（训练模式，内部计算均值/方差）

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    x: (N, C, D, H, W) numpy数组，输入张量
    scale: (C,) numpy数组，通道缩放 gamma
    offset: (C,) numpy数组，通道偏置 beta
    epsilon: float，数值稳定常数，默认 1e-5

    返回:
    y: (N, C, D, H, W) numpy数组，归一化输出（与 x 同 dtype）
    save_mean: (C,) float32 numpy数组，均值
    save_variance: (C,) float32 numpy数组，方差
    """
    orig_dtype = x.dtype
    N, C = x.shape[0], x.shape[1]

    x32 = x.astype(np.float32)
    scale32 = scale.astype(np.float32).reshape(1, C, 1, 1, 1)
    offset32 = offset.astype(np.float32).reshape(1, C, 1, 1, 1)

    # 沿 (N, D, H, W) 归约，每个通道独立统计
    mean = x32.mean(axis=(0, 2, 3, 4), keepdims=True)       # (1, C, 1, 1, 1)
    var = x32.var(axis=(0, 2, 3, 4), keepdims=True)         # (1, C, 1, 1, 1)

    y = (x32 - mean) / np.sqrt(var + epsilon) * scale32 + offset32

    if orig_dtype == np.float16:
        y_np = y.astype(np.float16)
    else:
        y_np = y.astype(np.float32)

    mean_np = mean.reshape(C).astype(np.float32)
    var_np = var.reshape(C).astype(np.float32)

    return y_np, mean_np, var_np


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
    input_x = np.ones((1, 4, 2, 2, 2)).astype(d_type_dict["float32"])
    input_scale = np.ones((4)).astype(d_type_dict["float32"])
    input_offset = np.ones((4)).astype(d_type_dict["float32"])
    attr_epsilon = 1e-05
    
    # 计算 golden 数据
    golden = impl(input_x, input_scale, input_offset, attr_epsilon)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_batch_norm3_d_x.bin")
    input_scale.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_batch_norm3_d_scale.bin")
    input_offset.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_batch_norm3_d_offset.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            _out_dtypes = ["float32", "float32", "float32"]
            for _gi, _g in enumerate(golden):
                _dt = _out_dtypes[_gi] if _gi < len(_out_dtypes) else _out_dtypes[-1]
                _g.astype(d_type_dict[_dt]).tofile(f"{_dt}_golden_batch_norm3_d_{_gi}.bin")
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_batch_norm3_d_0.bin")
    
    print(f"生成完成: dtype={d_type}")
