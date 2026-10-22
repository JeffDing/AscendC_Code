/*!
 * \file test_flash_attention.cpp
 * \brief FlashAttention 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "flash_attention_tiling.h"
#include "../../../op_kernel/flash_attention.cpp"

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

class FlashAttentionKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "FlashAttentionKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "FlashAttentionKernelTest TearDown" << endl;
    }
};

TEST_F(FlashAttentionKernelTest, test_kernel_run)
{
    constexpr size_t size = 256;
    constexpr size_t tilingDataSize = sizeof(FlashAttentionTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t qByteSize = 256 * 2;
    constexpr size_t kByteSize = 256 * 2;
    constexpr size_t vByteSize = 256 * 2;
    constexpr size_t outByteSize = 256 * 2;
    std::vector<float> qHost(256, 1);
    std::vector<float> kHost(256, 1);
    std::vector<float> vHost(256, 1);
    std::vector<float> outHost(256, 0);
    
    
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qByteSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(kByteSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(vByteSize);
    uint8_t* attn_mask = nullptr;
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 256; _i++) { uint16_t _h = FloatToHalf(qHost[_i]); memcpy(q + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 256; _i++) { uint16_t _h = FloatToHalf(kHost[_i]); memcpy(k + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 256; _i++) { uint16_t _h = FloatToHalf(vHost[_i]); memcpy(v + _i * 2, &_h, 2); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/flash_attention_tiling_data.h 中的字段定义
    //         参考 op_host/flash_attention_tiling.cpp 中的 tiling 计算逻辑
    FlashAttentionTilingData* tilingData = reinterpret_cast<FlashAttentionTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/flash_attention_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((flash_attention<0>), numBlocks, q, k, v, attn_mask, out, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(outHost.data(), out, outByteSize);
    { std::ofstream _ofs("float16_output_flash_attention_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(outHost.data()), outByteSize); }
    
    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(v);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
