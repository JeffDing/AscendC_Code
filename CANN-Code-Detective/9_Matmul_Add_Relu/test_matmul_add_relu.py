"""补全模板后，在 Ascend 950 上验证固定 256×256×256 的算子。"""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 Catlass Python MatmulAddRelu")
    parser.add_argument("--device", type=int, default=0, help="NPU 设备号，默认 0")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device 不能为负数")

    # 延迟导入，使 --help 在未安装 NPU 软件时也能使用。
    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack
    import torch
    import torch_npu  # noqa: F401，注册 torch.npu 后端。

    from matmul_add_relu import matmul_add_relu_kernel

    torch.npu.set_device(args.device)
    torch.manual_seed(2026)
    size = 256
    a = (torch.rand(size, size) * 2 - 1).half()
    b = (torch.rand(size, size) * 2 - 1).half()
    x = torch.linspace(-2.0, 2.0, size * size).reshape(size, size)
    expected = torch.relu(a.float() @ b.float() + x)

    device_a, device_b, device_x = a.npu(), b.npu(), x.npu()
    device_y = torch.full((size, size), -9999.0, dtype=torch.float32, device="npu")

    def wrap(tensor):
        return from_dlpack(
            tensor, layout_tag=tla.arch.RowMajor, origin_shape=(size, size)
        )

    ta, tb, tx, ty = map(wrap, (device_a, device_b, device_x, device_y))
    artifact = tla.compile(
        matmul_add_relu_kernel, ta, tb, tx, ty, options="--npu-arch 3510"
    )
    artifact(ta, tb, tx, ty, block_num=1)
    # 主机等待整个核函数完成，再读取结果。
    torch.npu.synchronize()

    actual = device_y.cpu()
    print(f"Catlass Python 计算结果：\n{actual}")
    print(f"PyTorch 参考结果：\n{expected}")
    tolerance = torch.maximum(torch.ones_like(expected), expected.abs()) / 256.0
    valid = torch.isfinite(actual) & (actual >= 0)
    valid &= (actual - expected).abs() <= tolerance
    if bool(valid.all()):
        print("PASS: M=N=K=256, blocks=1")
        return 0
    print(f"FAIL: M=N=K=256, mismatches={int((~valid).sum())}")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise
