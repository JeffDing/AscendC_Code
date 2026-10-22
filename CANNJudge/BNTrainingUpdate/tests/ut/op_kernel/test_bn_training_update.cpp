/*!
 * \file test_bn_training_update.cpp
 * \brief BnTrainingUpdate 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "bn_training_update_tiling.h"
#include "../../../op_kernel/bn_training_update.cpp"

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

class BnTrainingUpdateKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "BnTrainingUpdateKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "BnTrainingUpdateKernelTest TearDown" << endl;
    }
};

TEST_F(BnTrainingUpdateKernelTest, test_kernel_run)
{
    constexpr size_t size = 16;
    constexpr size_t tilingDataSize = sizeof(BnTrainingUpdateTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 16 * 4;
    constexpr size_t sumByteSize = 4 * 4;
    constexpr size_t square_sumByteSize = 4 * 4;
    constexpr size_t scaleByteSize = 4 * 4;
    constexpr size_t offsetByteSize = 4 * 4;
    constexpr size_t yByteSize = 16 * 4;
    constexpr size_t save_meanByteSize = 4 * 4;
    constexpr size_t save_varianceByteSize = 4 * 4;
    std::vector<float> xHost(16, 1);
    std::vector<float> sumHost(4, 1);
    std::vector<float> square_sumHost(4, 1);
    std::vector<float> scaleHost(4, 1);
    std::vector<float> offsetHost(4, 1);
    std::vector<float> yHost(16, 0);
    std::vector<float> save_meanHost(4, 0);
    std::vector<float> save_varianceHost(4, 0);
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* sum = (uint8_t*)AscendC::GmAlloc(sumByteSize);
    uint8_t* square_sum = (uint8_t*)AscendC::GmAlloc(square_sumByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(save_meanByteSize);
    uint8_t* save_variance = (uint8_t*)AscendC::GmAlloc(save_varianceByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x, xHost.data(), xByteSize);
    memcpy(sum, sumHost.data(), sumByteSize);
    memcpy(square_sum, square_sumHost.data(), square_sumByteSize);
    memcpy(scale, scaleHost.data(), scaleByteSize);
    memcpy(offset, offsetHost.data(), offsetByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/bn_training_update_tiling_data.h 中的字段定义
    //         参考 op_host/bn_training_update_tiling.cpp 中的 tiling 计算逻辑
    BnTrainingUpdateTilingData* tilingData = reinterpret_cast<BnTrainingUpdateTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/bn_training_update_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((bn_training_update<1>), numBlocks, x, sum, square_sum, scale, offset, y, save_mean, save_variance, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_bn_training_update_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    memcpy(save_meanHost.data(), save_mean, save_meanByteSize);
    { std::ofstream _ofs("float32_output_bn_training_update_1.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(save_meanHost.data()), save_meanByteSize); }
    memcpy(save_varianceHost.data(), save_variance, save_varianceByteSize);
    { std::ofstream _ofs("float32_output_bn_training_update_2.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(save_varianceHost.data()), save_varianceByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(sum);
    AscendC::GmFree(square_sum);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_variance);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
