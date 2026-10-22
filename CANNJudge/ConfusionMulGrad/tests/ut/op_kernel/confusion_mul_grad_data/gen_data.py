#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch

_NDIM = 2


def impl(input0, input1, input2, axis, keep_dims=False):
    """ConfusionMulGrad算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
        input0:   (M, N) numpy 数组或 torch.Tensor，逐元素乘的左操作数
        input1:   (M, N) 与 input0 同 shape 同 dtype，共享操作数
        input2:   (M, N) 与 input0 同 shape 同 dtype，归约乘的右操作数
        axis:     list_int，归约轴，取值限于 [1]/[0]/[0,1]
        keep_dims: bool，默认 False，归约后是否保留长度为 1 的维度

    返回与输入同为 numpy 或同为 torch 的输出:
        output0: (M, N) 逐元素乘结果，dtype 与输入一致
        output1: ReduceSum(input1 ⊙ input2, axis, keep_dims)，dtype 与输入一致
    """
    return_numpy = not isinstance(input0, torch.Tensor)
    a = torch.as_tensor(input0)         # numpy / torch 输入统一转 torch
    b = torch.as_tensor(input1)
    c = torch.as_tensor(input2)
    orig_dtype = a.dtype

    # 归约轴归一化（负轴按 2D 语义）并去重排序
    if isinstance(axis, (int, np.integer)):
        axis = [axis]
    dims = sorted({(int(v) + _NDIM) % _NDIM for v in axis})

    # 统一 FLOAT32 计算（fp16 输入先升精度，归约累加红线）
    a32, b32, c32 = a.float(), b.float(), c.float()

    # output0 = input0 ⊙ input1（FLOAT32 计算后 cast 回输入 dtype）
    out0 = (a32 * b32).to(orig_dtype)

    # output1 = ReduceSum(input1 ⊙ input2, axis, keep_dims)，FLOAT32 累加
    out1 = (b32 * c32).sum(dim=dims, keepdim=bool(keep_dims)).to(orig_dtype)

    if return_numpy:
        return out0.numpy(), out1.numpy()
    return out0, out1


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
    input_input0 = np.ones((3, 64)).astype(d_type_dict["float32"])
    input_input1 = np.ones((3, 64)).astype(d_type_dict["float32"])
    input_input2 = np.ones((3, 64)).astype(d_type_dict["float32"])
    attr_axis = [1]
    attr_keep_dims = False
    
    # 计算 golden 数据
    golden = impl(input_input0, input_input1, input_input2, attr_axis, attr_keep_dims)
    
    # 保存数据到文件
    input_input0.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_confusion_mul_grad_input0.bin")
    input_input1.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_confusion_mul_grad_input1.bin")
    input_input2.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_confusion_mul_grad_input2.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            _out_dtypes = ["float32", "float32"]
            for _gi, _g in enumerate(golden):
                _dt = _out_dtypes[_gi] if _gi < len(_out_dtypes) else _out_dtypes[-1]
                _g.astype(d_type_dict[_dt]).tofile(f"{_dt}_golden_confusion_mul_grad_{_gi}.bin")
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_confusion_mul_grad_0.bin")
    
    print(f"生成完成: dtype={d_type}")
