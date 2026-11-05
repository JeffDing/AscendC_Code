#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(x, w, b=None, stride=1, pad=0, dilation=1):
    """DepthwiseConv2D算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    x: (N, C, H, W) numpy数组，输入张量
    w: (C, 1, kH, kW) numpy数组，深度卷积权重（与 x 同 dtype）
    b: (C,) numpy数组或 None，per-channel 偏置，默认 None（不加偏置）
    stride: int，滑动步长，默认 1
    pad: int，空间零填充量，默认 0
    dilation: int，膨胀率，默认 1

    返回:
    y: (N, C, H_out, W_out) numpy数组，卷积输出（与 x 同 dtype）
    """
    N, C, H, W = x.shape
    _, _, kH, kW = w.shape

    if pad > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    else:
        x_padded = x

    H_out = (H + 2 * pad - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * pad - dilation * (kW - 1) - 1) // stride + 1

    x32 = x_padded.astype(np.float32)
    w32 = w.astype(np.float32)

    y = np.zeros((N, C, H_out, W_out), dtype=np.float32)

    for kh in range(kH):
        for kw in range(kW):
            hs = kh * dilation
            ws = kw * dilation
            patch = x32[:, :, hs:hs + (H_out - 1) * stride + 1:stride,
                               ws:ws + (W_out - 1) * stride + 1:stride]
            y += patch * w32[:, 0, kh, kw].reshape(1, C, 1, 1)

    if b is not None:
        y += b.astype(np.float32).reshape(1, C, 1, 1)

    if x.dtype == np.float16:
        y_np = y.astype(np.float16)
    else:
        y_np = y.astype(np.float32)

    return y_np


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
    input_x = np.ones((1, 4, 8, 8)).astype(d_type_dict["float32"])
    input_w = np.ones((4, 1, 3, 3)).astype(d_type_dict["float32"])
    input_b = np.ones((4)).astype(d_type_dict["float32"])
    attr_stride = 1
    attr_pad = 1
    attr_dilation = 1
    
    # 计算 golden 数据
    golden = impl(input_x, input_w, input_b, attr_stride, attr_pad, attr_dilation)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_depthwise_conv2_d_x.bin")
    input_w.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_depthwise_conv2_d_w.bin")
    input_b.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_depthwise_conv2_d_b.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_depthwise_conv2_d.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_depthwise_conv2_d.bin")
    
    print(f"生成完成: dtype={d_type}")
