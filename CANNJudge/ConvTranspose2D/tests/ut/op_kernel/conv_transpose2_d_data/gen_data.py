#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch
import torch.nn.functional as F


def _pair(value):
    """int / (int, int) 统一成 (h, w) 二元组"""
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def impl(x, w, b=None, stride=1, padding=0, output_padding=0, dilation=1):
    """ConvTranspose2D算子golden实现（groups = 1）

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
        x: (N, C_in, H_in, W_in) numpy 数组或 torch.Tensor，输入张量
        w: (C_in, C_out, kH, kW) 转置卷积权重（与 x 同 dtype）
        b: (C_out,) 输出通道偏置（float32）或 None，默认 None（不叠加偏置）
        stride: int，输入元素在输出上的滑动步长（上采样倍率），默认 1
        padding: int，输出侧裁剪量，默认 0
        output_padding: int，输出末端额外零填充量，默认 0
        dilation: int，卷积核膨胀率，默认 1

    返回与输入同为 numpy 或同为 torch 的输出:
        y: (N, C_out, H_out, W_out)，dtype 与 x 一致
    """
    return_numpy = not isinstance(x, torch.Tensor)
    x_t = torch.as_tensor(x)            # numpy / torch 输入统一转 torch
    orig_dtype = x_t.dtype
    w_t = torch.as_tensor(w)

    # 统一 FLOAT32 计算（fp16 精度不足，内部用 FP32）；偏置固定 fp32
    x32 = x_t.float()
    w32 = w_t.float()
    b32 = None
    if b is not None:
        b_t = torch.as_tensor(b)
        if b_t.numel() > 0:
            b32 = b_t.float()

    # 转置卷积：等价于插零上采样后步长 1 滑窗加权求和，再按 padding 裁剪并叠偏置
    y = F.conv_transpose2d(
        x32, w32, b32,
        stride=_pair(stride), padding=_pair(padding),
        output_padding=_pair(output_padding), dilation=_pair(dilation),
    )

    # 转回原始 dtype（与 JSON output_desc 一致）
    y = y.to(orig_dtype)

    if return_numpy:
        return y.numpy()
    return y


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
    input_w = np.ones((4, 4, 3, 3)).astype(d_type_dict["float32"])
    input_b = np.ones((4)).astype(d_type_dict["float32"])
    attr_stride = 1
    attr_padding = 0
    attr_output_padding = 0
    attr_dilation = 1
    
    # 计算 golden 数据
    golden = impl(input_x, input_w, input_b, attr_stride, attr_padding, attr_output_padding, attr_dilation)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_conv_transpose2_d_x.bin")
    input_w.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_conv_transpose2_d_w.bin")
    input_b.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_conv_transpose2_d_b.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_conv_transpose2_d.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_conv_transpose2_d.bin")
    
    print(f"生成完成: dtype={d_type}")
