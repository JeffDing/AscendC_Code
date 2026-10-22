/*!
 * \file test_confusion_mul_grad.cpp
 * \brief ConfusionMulGrad 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "confusion_mul_grad_tiling.h"
#include "../../../op_kernel/confusion_mul_grad.cpp"

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

class ConfusionMulGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ConfusionMulGradKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ConfusionMulGradKernelTest TearDown" << endl;
    }
};

TEST_F(ConfusionMulGradKernelTest, test_kernel_run)
{
    constexpr size_t size = 192;
    constexpr size_t tilingDataSize = sizeof(ConfusionMulGradTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t input0ByteSize = 192 * 4;
    constexpr size_t input1ByteSize = 192 * 4;
    constexpr size_t input2ByteSize = 192 * 4;
    constexpr size_t output0ByteSize = 192 * 4;
    constexpr size_t output1ByteSize = 3 * 4;
    std::vector<float> input0Host(192, 1);
    std::vector<float> input1Host(192, 1);
    std::vector<float> input2Host(192, 1);
    std::vector<float> output0Host(192, 0);
    std::vector<float> output1Host(3, 0);
    
    
    uint8_t* input0 = (uint8_t*)AscendC::GmAlloc(input0ByteSize);
    uint8_t* input1 = (uint8_t*)AscendC::GmAlloc(input1ByteSize);
    uint8_t* input2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* output0 = (uint8_t*)AscendC::GmAlloc(output0ByteSize);
    uint8_t* output1 = (uint8_t*)AscendC::GmAlloc(output1ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(input0, input0Host.data(), input0ByteSize);
    memcpy(input1, input1Host.data(), input1ByteSize);
    memcpy(input2, input2Host.data(), input2ByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/confusion_mul_grad_tiling_data.h 中的字段定义
    //         参考 op_host/confusion_mul_grad_tiling.cpp 中的 tiling 计算逻辑
    ConfusionMulGradTilingData* tilingData = reinterpret_cast<ConfusionMulGradTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/confusion_mul_grad_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((confusion_mul_grad<1>), numBlocks, input0, input1, input2, output0, output1, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(output0Host.data(), output0, output0ByteSize);
    { std::ofstream _ofs("float32_output_confusion_mul_grad_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(output0Host.data()), output0ByteSize); }
    memcpy(output1Host.data(), output1, output1ByteSize);
    { std::ofstream _ofs("float32_output_confusion_mul_grad_1.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(output1Host.data()), output1ByteSize); }
    
    AscendC::GmFree(input0);
    AscendC::GmFree(input1);
    AscendC::GmFree(input2);
    AscendC::GmFree(output0);
    AscendC::GmFree(output1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
