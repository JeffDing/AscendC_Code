/*!
 * \file test_quant_matmul_sigmoid_mul_quant.cpp
 * \brief QuantMatmulSigmoidMulQuant 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "quant_matmul_sigmoid_mul_quant_tiling.h"
#include "../../../op_kernel/quant_matmul_sigmoid_mul_quant.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class QuantMatmulSigmoidMulQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "QuantMatmulSigmoidMulQuantKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "QuantMatmulSigmoidMulQuantKernelTest TearDown" << endl;
    }
};

TEST_F(QuantMatmulSigmoidMulQuantKernelTest, test_kernel_run)
{
    constexpr size_t size = 16;
    constexpr size_t tilingDataSize = sizeof(QuantMatmulSigmoidMulQuantTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t x1ByteSize = 16 * 1;
    constexpr size_t x2ByteSize = 16 * 1;
    constexpr size_t x1ScaleByteSize = 1 * 4;
    constexpr size_t x2ScaleByteSize = 1 * 4;
    constexpr size_t yByteSize = 1 * 1;
    constexpr size_t yScaleByteSize = 1 * 4;
    std::vector<int8_t> x1Host(16, 1);
    std::vector<int8_t> x2Host(16, 1);
    std::vector<float> x1ScaleHost(1, 1);
    std::vector<float> x2ScaleHost(1, 1);
    std::vector<int8_t> yHost(1, 0);
    std::vector<float> yScaleHost(1, 0);
    
    
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* x1Scale = (uint8_t*)AscendC::GmAlloc(x1ScaleByteSize);
    uint8_t* x2Scale = (uint8_t*)AscendC::GmAlloc(x2ScaleByteSize);
    uint8_t* bias = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* yScale = (uint8_t*)AscendC::GmAlloc(yScaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x1, x1Host.data(), x1ByteSize);
    memcpy(x2, x2Host.data(), x2ByteSize);
    memcpy(x1Scale, x1ScaleHost.data(), x1ScaleByteSize);
    memcpy(x2Scale, x2ScaleHost.data(), x2ScaleByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/quant_matmul_sigmoid_mul_quant_tiling_data.h 中的字段定义
    //         参考 op_host/quant_matmul_sigmoid_mul_quant_tiling.cpp 中的 tiling 计算逻辑
    QuantMatmulSigmoidMulQuantTilingData* tilingData = reinterpret_cast<QuantMatmulSigmoidMulQuantTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/quant_matmul_sigmoid_mul_quant_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((quant_matmul_sigmoid_mul_quant<1>), numBlocks, x1, x2, x1Scale, x2Scale, bias, y, yScale, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("int8_output_quant_matmul_sigmoid_mul_quant_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    memcpy(yScaleHost.data(), yScale, yScaleByteSize);
    { std::ofstream _ofs("float32_output_quant_matmul_sigmoid_mul_quant_1.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yScaleHost.data()), yScaleByteSize); }
    
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(x1Scale);
    AscendC::GmFree(x2Scale);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
