#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(self, target, reduction=1):
    """MseLoss算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    self: numpy数组，预测值 x
    target: numpy数组，目标值 y，与 self 同 shape
    reduction: int，缩减方式，0=none, 1=mean, 2=sum，默认 1

    返回:
    out: reduction=0 时与 self 同 shape；否则为 shape [1] 的标量数组
    """
    orig_dtype = self.dtype
    x32 = self.astype(np.float32)
    y32 = target.astype(np.float32)

    l = (x32 - y32) ** 2

    if reduction == 0:
        out32 = l
    elif reduction == 1:
        out32 = np.array([l.mean()], dtype=np.float32)
    else:  # reduction == 2
        out32 = np.array([l.sum()], dtype=np.float32)

    if orig_dtype == np.float16:
        return out32.astype(np.float16)
    return out32.astype(np.float32)


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
    input_self = np.ones((8)).astype(d_type_dict["float32"])
    input_target = np.ones((8)).astype(d_type_dict["float32"])
    attr_reduction = 1
    
    # 计算 golden 数据
    golden = impl(input_self, input_target, attr_reduction)
    
    # 保存数据到文件
    input_self.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_mse_loss_self.bin")
    input_target.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_mse_loss_target.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_mse_loss.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_mse_loss.bin")
    
    print(f"生成完成: dtype={d_type}")
