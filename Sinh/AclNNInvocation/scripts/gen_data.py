#!/usr/bin/python3 
# -*- coding:utf-8__  
# 版权所有 (c) 华为技术有限公司 2022-2023。

import numpy as np  # 导入numpy库，用于科学计算

# 定义生成基准数据的函数
def gen_golden_data_simple():
    # 生成一个大小为[8, 2048]的随机数矩阵，范围在[-100, 100]之间，数据类型为float16
    input_x = np.random.uniform(-100, 100, [8, 2048]).astype(np.float16)
    
    # 计算两个矩阵的和，得到基准数据（golden data），数据类型为float16
    golden = np.sinh(input_x).astype(np.float16)
    
    # 将生成的输入矩阵input_x和input_y分别保存到二进制文件"./input/input_x.bin"和"./input/input_y.bin"
    # 将基准数据golden保存到二进制文件"./output/golden.bin"，用于后续结果验证
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    # 调用函数生成基准数据
    gen_golden_data_simple()

