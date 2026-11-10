import numpy as np

def impl(x, negative_slope=0.01):
    # 验算函数：逐元素 leaky relu，y = max(x, 0) + negative_slope * min(x, 0)
    # float64 中间计算，最后一次性转回输入 dtype
    dtype = x.dtype
    x64 = x.astype(np.float64)
    y = np.where(x64 > 0, x64, np.float64(negative_slope) * x64)
    return y.astype(dtype)

if __name__ == "__main__":
    for dtype in (np.float16, np.float32):
        x = np.random.uniform(-10, 10, (8, 2)).astype(dtype)
        y = impl(x, 0.01)
        x64 = x.astype(np.float64)
        ref = np.where(x64 > 0, x64, 0.01 * x64).astype(dtype)
        ok = np.array_equal(y, ref)
        print("dtype={}, shape={}, output dtype={}, match={}".format(
            np.dtype(dtype).name, x.shape, y.dtype, ok))

