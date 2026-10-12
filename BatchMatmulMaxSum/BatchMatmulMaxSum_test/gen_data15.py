#!/usr/bin/env python3
"""Generate the 15 official OJ cases (x1/x2 + FP64 golden y) locally.

Self-contained: no ml_dtypes needed (bf16 quantised by hand, matching the
official generator: round-to-nearest-even via +0x8000 >> 16).

Seeds and shapes are identical to the official harness (20260928 + caseId,
uniform [-1, 1]).
"""
import os
import sys

import numpy as np

# (id, B, M, N, K, transposeX1, transposeX2, dtype)  dtype: 1=fp16 2=bf16
CASES = [
    (1,  1,    1,    1,   32,  False, False, 1),
    (2,  1,    3,    5,   40,  True,  False, 2),
    (3,  2,    7,   13,   48,  False, True,  1),
    (4,  4,    8,   16,  128,  True,  True,  2),
    (5,  8,   16,   32,  128,  False, False, 1),
    (6,  13,   23,   73,  192, False, True,  2),
    (7,  16,   32,  128,  256, True,  False, 1),
    (8,  1,  1025, 1031, 1032, True,  True,  2),
    (9,  1,  2048, 1536, 1280, False, True,  1),
    (10, 1,  4096, 1280, 1152, False, False, 2),
    (11, 1,  1536, 2048, 2048, True,  True,  1),
    (12, 1,  1280, 4096, 1536, True,  False, 2),
    (13, 1,  8192,   64,  128, False, False, 1),
    (14, 1,    64, 8192,  128, False, True,  2),
    (15, 1,    64,   64, 8192, True,  True,  1),
]


def quantize(values, dtype):
    """Return (packed uint16, logical float32) with official rounding."""
    values = values.astype(np.float32, copy=False)
    if dtype == 1:
        packed = values.astype(np.float16)
        return packed.view(np.uint16), packed.astype(np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)
    restored = (rounded.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return rounded, restored


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(os.path.join(root, "input"), exist_ok=True)
    os.makedirs(os.path.join(root, "output"), exist_ok=True)
    for cid, B, M, N, K, tx1, tx2, dtype in CASES:
        rng = np.random.default_rng(20260928 + cid)
        a = rng.uniform(-1.0, 1.0, (B, M, K)).astype(np.float32)
        b = rng.uniform(-1.0, 1.0, (B, K, N)).astype(np.float32)
        pa, la = quantize(a, dtype)
        pb, lb = quantize(b, dtype)
        # physical storage layout
        sa = np.transpose(pa, (0, 2, 1)) if tx1 else pa      # (B,K,M) when tx1
        sb = np.transpose(pb, (0, 2, 1)) if tx2 else pb      # (B,N,K) when tx2
        cdir = os.path.join(root, "input", f"case{cid}")
        os.makedirs(cdir, exist_ok=True)
        np.ascontiguousarray(sa).astype(np.uint16).tofile(os.path.join(cdir, "x1.bin"))
        np.ascontiguousarray(sb).astype(np.uint16).tofile(os.path.join(cdir, "x2.bin"))
        # FP64 golden on logical tensors (la/lb already logical (B,M,K)/(B,K,N))
        products = np.matmul(la.astype(np.float64), lb.astype(np.float64))
        y = np.sum(np.max(products, axis=-1), axis=-1).astype(np.float32)
        gdir = os.path.join(root, "output", f"golden_case{cid}")
        os.makedirs(gdir, exist_ok=True)
        y.tofile(os.path.join(gdir, "golden_y.bin"))
        print(f"case{cid:2d}: B={B} M={M} N={N} K={K} tx=({int(tx1)},{int(tx2)}) "
              f"dtype={'fp16' if dtype == 1 else 'bf16'} golden[0]={y[0]:.6f}")


if __name__ == "__main__":
    main()
