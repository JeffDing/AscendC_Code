/*!
 * \file test_mhc_sinkhorn_grad.cpp
 * \brief MhcSinkhornGrad 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "mhc_sinkhorn_grad_tiling.h"
#include "../../../op_kernel/mhc_sinkhorn_grad.cpp"

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

class MhcSinkhornGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcSinkhornGradKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MhcSinkhornGradKernelTest TearDown" << endl;
    }
};

TEST_F(MhcSinkhornGradKernelTest, test_kernel_run)
{
    constexpr size_t size = 16;
    constexpr size_t tilingDataSize = sizeof(MhcSinkhornGradTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t gradOutputByteSize = 16 * 4;
    constexpr size_t normOutByteSize = 32 * 4;
    constexpr size_t sumOutByteSize = 8 * 4;
    constexpr size_t gradInputByteSize = 16 * 4;
    std::vector<float> gradOutputHost(16, 1);
    std::vector<float> normOutHost(32, 1);
    std::vector<float> sumOutHost(8, 1);
    std::vector<float> gradInputHost(16, 0);
    
    
    uint8_t* gradOutput = (uint8_t*)AscendC::GmAlloc(gradOutputByteSize);
    uint8_t* normOut = (uint8_t*)AscendC::GmAlloc(normOutByteSize);
    uint8_t* sumOut = (uint8_t*)AscendC::GmAlloc(sumOutByteSize);
    uint8_t* gradInput = (uint8_t*)AscendC::GmAlloc(gradInputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(gradOutput, gradOutputHost.data(), gradOutputByteSize);
    memcpy(normOut, normOutHost.data(), normOutByteSize);
    memcpy(sumOut, sumOutHost.data(), sumOutByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/mhc_sinkhorn_grad_tiling_data.h 中的字段定义
    //         参考 op_host/mhc_sinkhorn_grad_tiling.cpp 中的 tiling 计算逻辑
    MhcSinkhornGradTilingData* tilingData = reinterpret_cast<MhcSinkhornGradTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/mhc_sinkhorn_grad_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((mhc_sinkhorn_grad<1>), numBlocks, gradOutput, normOut, sumOut, gradInput, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(gradInputHost.data(), gradInput, gradInputByteSize);
    { std::ofstream _ofs("float32_output_mhc_sinkhorn_grad_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(gradInputHost.data()), gradInputByteSize); }
    
    AscendC::GmFree(gradOutput);
    AscendC::GmFree(normOut);
    AscendC::GmFree(sumOut);
    AscendC::GmFree(gradInput);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
