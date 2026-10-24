/*!
 * \file test_mse_loss.cpp
 * \brief MseLoss 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "mse_loss_tiling.h"
#include "../../../op_kernel/mse_loss.cpp"

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

class MseLossKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MseLossKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MseLossKernelTest TearDown" << endl;
    }
};

TEST_F(MseLossKernelTest, test_kernel_run)
{
    constexpr size_t size = 8;
    constexpr size_t tilingDataSize = sizeof(MseLossTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t selfByteSize = 8 * 4;
    constexpr size_t targetByteSize = 8 * 4;
    constexpr size_t outByteSize = 1 * 4;
    std::vector<float> selfHost(8, 1);
    std::vector<float> targetHost(8, 1);
    std::vector<float> outHost(1, 0);
    
    
    uint8_t* self = (uint8_t*)AscendC::GmAlloc(selfByteSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(self, selfHost.data(), selfByteSize);
    memcpy(target, targetHost.data(), targetByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/mse_loss_tiling_data.h 中的字段定义
    //         参考 op_host/mse_loss_tiling.cpp 中的 tiling 计算逻辑
    MseLossTilingData* tilingData = reinterpret_cast<MseLossTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/mse_loss_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((mse_loss<1>), numBlocks, self, target, out, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(outHost.data(), out, outByteSize);
    { std::ofstream _ofs("float32_output_mse_loss_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(outHost.data()), outByteSize); }
    
    AscendC::GmFree(self);
    AscendC::GmFree(target);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
