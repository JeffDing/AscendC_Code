#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch

NEG_MASK = -1e4  # 因果掩码加性大负值（避免 -inf 引发 NaN 的硬件友好口径）


def impl(q, k, v, attn_mask, softmax_scale, is_causal):
    """FlashAttention算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
        q: (B, N, S, D) numpy 数组或 torch.Tensor，查询张量
        k: (B, N, S, D) 键张量
        v: (B, N, S, D) 值张量
        attn_mask: (B, 1, S, S) 加性掩码或 None（JSON 用空数组表示未提供）
        softmax_scale: float，缩放因子，<= 0 表示使用默认值 1/sqrt(D)
        is_causal: bool，是否应用因果掩码

    返回与输入同为 numpy 或同为 torch 的输出:
        out: (B, N, S, D) 注意力输出，dtype 与 q 一致
    """
    return_numpy = not isinstance(q, torch.Tensor)
    q_t = torch.as_tensor(q)            # numpy / torch 输入统一转 torch
    k_t = torch.as_tensor(k)
    v_t = torch.as_tensor(v)
    orig_dtype = q_t.dtype

    # 统一转 float32 计算（fp16/bf16 精度不足，内部用 FP32）
    q32, k32, v32 = q_t.float(), k_t.float(), v_t.float()
    B, N, S, D = q32.shape

    # 缩放因子：未提供（None）或 <= 0 时取默认值 1/sqrt(D)
    scale = float(softmax_scale) if softmax_scale is not None else 0.0
    if scale <= 0.0:
        scale = 1.0 / float(D) ** 0.5

    # 缩放点积 -> (B, N, S, S)
    scores = torch.matmul(q32, k32.transpose(-1, -2)) * scale

    # 因果掩码：屏蔽上三角（j > i），加性大负值
    if is_causal:
        causal = torch.triu(torch.full((S, S), NEG_MASK, dtype=torch.float32), diagonal=1)
        scores = scores + causal

    # 加性掩码（可选输入；未提供时空数组跳过）
    if attn_mask is not None:
        mask = torch.as_tensor(attn_mask)
        if mask.numel() > 0:
            scores = scores + mask.float()

    # 数值稳定 softmax（沿最后一维，减去行最大值）
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.exp(scores)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # 加权求和 -> (B, N, S, D)
    out = torch.matmul(probs, v32).to(orig_dtype)

    if return_numpy:
        return out.numpy()
    return out


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
    input_q = np.ones((1, 2, 8, 16)).astype(d_type_dict["float16"])
    input_k = np.ones((1, 2, 8, 16)).astype(d_type_dict["float16"])
    input_v = np.ones((1, 2, 8, 16)).astype(d_type_dict["float16"])
    input_attn_mask = None  # 可选输入，未提供
    attr_softmax_scale = 0.0
    attr_is_causal = True
    
    # 计算 golden 数据
    golden = impl(input_q, input_k, input_v, input_attn_mask, attr_softmax_scale, attr_is_causal)
    
    # 保存数据到文件
    input_q.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_flash_attention_q.bin")
    input_k.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_flash_attention_k.bin")
    input_v.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_flash_attention_v.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float16_golden_flash_attention.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float16"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float16"]).tofile("float16_golden_flash_attention.bin")
    
    print(f"生成完成: dtype={d_type}")
