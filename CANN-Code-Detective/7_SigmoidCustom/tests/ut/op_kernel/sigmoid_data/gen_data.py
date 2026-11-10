#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(x):
    # 验算函数：逐元素 sigmoid，float64 中间计算，最后一次性转回输入 dtype
    dtype = x.dtype
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(dtype)


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
    input_x = np.ones((8, 2048)).astype(d_type_dict["float32"])

    
    # 计算 golden 数据
    golden = impl(input_x)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_sigmoid_x.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_sigmoid.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_sigmoid.bin")
    
    print(f"生成完成: dtype={d_type}")
