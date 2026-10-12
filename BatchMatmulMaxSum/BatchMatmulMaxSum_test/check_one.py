#!/usr/bin/env python3
"""Compare output/y.bin against the FP64 golden for one case (wasz criterion:
rtol=1e-3, atol=1e-3; report max_rel)."""
import os
import sys

import numpy as np

cid = int(sys.argv[1])
root = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))

golden = np.fromfile(f"{root}/output/golden_case{cid}/golden_y.bin", dtype=np.float32)
out = np.fromfile(f"{root}/output/y.bin", dtype=np.float32)
if out.size < golden.size:
    print(f"case{cid}: FAIL short output {out.size} < {golden.size}")
    sys.exit(1)
got = out[:golden.size].astype(np.float64)
ref = golden.astype(np.float64)
rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)))
if np.allclose(got, ref, rtol=1e-3, atol=1e-3):
    print(f"case{cid}: PASS (y[0]={got[0]:.4f}, max_rel={rel:.2e})")
    sys.exit(0)
print(f"case{cid}: FAIL max_rel={rel:.3e}\n  out   ={got}\n  golden={ref}")
sys.exit(1)
