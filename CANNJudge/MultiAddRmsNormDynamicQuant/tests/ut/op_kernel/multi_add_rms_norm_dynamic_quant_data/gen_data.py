#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch

_INT8_LOW = -128
_INT8_HIGH = 127


def _as_torch(x):
    """numpy（含 ml_dtypes bfloat16）/ torch 输入统一转 torch.Tensor。"""
    if isinstance(x, torch.Tensor):
        return x
    arr = np.ascontiguousarray(x)
    if arr.dtype.name == "bfloat16":    # numpy 无原生 bf16，按位视图重建
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    return torch.as_tensor(arr)


def _to_numpy(t):
    """torch 输出转 numpy（numpy 无原生 bfloat16，经 ml_dtypes 转换）。"""
    if t.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16
        return t.float().numpy().astype(bfloat16)
    return t.numpy()


def impl(x1a, x1b, x1c, x1d, x1e, x2, gamma, smoothScale1, smoothScale2,
         epsilon=1e-6):
    """MultiAddRmsNormDynamicQuant算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
        x1a: (T, H) numpy 数组或 torch.Tensor，add 列表首项（float16/bfloat16）
        x1b~x1e: (T, H) 可选追加 add 项，None（或空数组）表示未输入
        x2: (T, H) 残差项，与 x1a 同 dtype 同 shape
        gamma: (H,) RMSNorm 权重
        smoothScale1/smoothScale2: (H,) 两路量化的平滑权重
        epsilon: float，防除 0 的小常数，默认 1e-6，加在 mean(x^2) 之后

    返回与输入同为 numpy 或同为 torch 的输出:
        y1/y2: (T, H) int8 量化结果；x/y: (T, H) 与输入同 dtype；
        scale1/scale2: (T,) float32 逐行量化 scale。
    """
    return_numpy = not isinstance(x1a, torch.Tensor)
    a = _as_torch(x1a)                  # numpy / torch 输入统一转 torch
    orig_dtype = a.dtype

    # 1. 多路 add，全程 FLOAT32 累加（先 x1a + x2，再依次累加缺省后的追加项）
    x32 = a.float() + _as_torch(x2).float()
    for extra in (x1b, x1c, x1d, x1e):
        if extra is not None and np.size(extra) > 0:
            x32 = x32 + _as_torch(extra).float()

    # 2. 逐行 RmsNorm（FLOAT32）：rstd = 1/sqrt(mean(x^2)+eps)，z = x*rstd*gamma
    var = (x32 * x32).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + float(epsilon))
    z = x32 * rstd * _as_torch(gamma).float()

    # 3. x/y 输出 cast 回输入 dtype（量化输入仍是 FLOAT32 的 z）
    x_out = x32.to(orig_dtype)
    y_out = z.to(orig_dtype)

    def quant(z_src, smooth):
        q = z_src * _as_torch(smooth).float()
        scale = q.abs().amax(dim=-1, keepdim=True) / 127.0
        y_q = torch.clamp(torch.round(q / scale), _INT8_LOW, _INT8_HIGH).to(torch.int8)
        return y_q, scale[:, 0]

    # 4. 双路 per-row 动态量化（rint 为 round-to-nearest-even）
    y1, scale1 = quant(z, smoothScale1)
    y2, scale2 = quant(z, smoothScale2)

    if return_numpy:
        return (_to_numpy(y1), _to_numpy(y2), _to_numpy(x_out), _to_numpy(y_out),
                _to_numpy(scale1), _to_numpy(scale2))
    return y1, y2, x_out, y_out, scale1, scale2


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float16"
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
    input_x1a = np.ones((1, 64)).astype(d_type_dict["float16"])
    input_x1b = np.ones((1, 64)).astype(d_type_dict["float16"])
    input_x1c = np.ones((1, 64)).astype(d_type_dict["float16"])
    input_x1d = np.ones((1, 64)).astype(d_type_dict["float16"])
    input_x1e = np.ones((1, 64)).astype(d_type_dict["float16"])
    input_x2 = np.ones((1, 64)).astype(d_type_dict["float16"])
    input_gamma = np.ones((64)).astype(d_type_dict["float16"])
    input_smoothScale1 = np.ones((64)).astype(d_type_dict["float16"])
    input_smoothScale2 = np.ones((64)).astype(d_type_dict["float16"])
    attr_epsilon = 1e-06
    
    # 计算 golden 数据
    golden = impl(input_x1a, input_x1b, input_x1c, input_x1d, input_x1e, input_x2, input_gamma, input_smoothScale1, input_smoothScale2, attr_epsilon)
    
    # 保存数据到文件
    input_x1a.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_x1a.bin")
    input_x1b.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_x1b.bin")
    input_x1c.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_x1c.bin")
    input_x1d.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_x1d.bin")
    input_x1e.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_x1e.bin")
    input_x2.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_x2.bin")
    input_gamma.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_gamma.bin")
    input_smoothScale1.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_smoothScale1.bin")
    input_smoothScale2.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_multi_add_rms_norm_dynamic_quant_smoothScale2.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            _out_dtypes = ["int8", "int8", "float16", "float16", "float32", "float32"]
            for _gi, _g in enumerate(golden):
                _dt = _out_dtypes[_gi] if _gi < len(_out_dtypes) else _out_dtypes[-1]
                _g.astype(d_type_dict[_dt]).tofile(f"{_dt}_golden_multi_add_rms_norm_dynamic_quant_{_gi}.bin")
        else:
            golden.astype(d_type_dict["int8"]).tofile("int8_golden_multi_add_rms_norm_dynamic_quant_0.bin")
    
    print(f"生成完成: dtype={d_type}")
