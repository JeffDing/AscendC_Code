#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch

def impl(x, mask, indices):
    x_torch = torch.from_numpy(x)
    indices_torch = torch.from_numpy(np.array(indices))
    
    cmd = "x_torch["
    idx = 0
    for i in range(mask.size):
        if mask[i]:
            cmd += "indices_torch"
            idx += 1
        else:
            cmd += ":"
        if i < mask.size - 1:
            cmd += ","
    cmd += "]"
    
    res = eval(cmd)
    res = res.numpy()
    return res


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
    input_x = np.ones((8192, 128)).astype(d_type_dict["float32"])
    input_indexed_sizes = np.ones((2)).astype(d_type_dict["int64"])
    input_indices = np.ones((10)).astype(d_type_dict["int64"])

    
    # 计算 golden 数据
    golden = impl(input_x, input_indexed_sizes, input_indices)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_index_x.bin")
    input_indexed_sizes.astype(d_type_dict["int64"]).tofile(f"{d_type}_input_index_indexed_sizes.bin")
    input_indices.astype(d_type_dict["int64"]).tofile(f"{d_type}_input_index_indices.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_index.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_index.bin")
    
    print(f"生成完成: dtype={d_type}")
