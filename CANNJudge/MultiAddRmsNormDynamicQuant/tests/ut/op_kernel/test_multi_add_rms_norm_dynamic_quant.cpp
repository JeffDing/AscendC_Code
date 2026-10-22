/*!
 * \file test_multi_add_rms_norm_dynamic_quant.cpp
 * \brief MultiAddRmsNormDynamicQuant 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "multi_add_rms_norm_dynamic_quant_tiling.h"
#include "../../../op_kernel/multi_add_rms_norm_dynamic_quant.cpp"

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

class MultiAddRmsNormDynamicQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MultiAddRmsNormDynamicQuantKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MultiAddRmsNormDynamicQuantKernelTest TearDown" << endl;
    }
};

TEST_F(MultiAddRmsNormDynamicQuantKernelTest, test_kernel_run)
{
    constexpr size_t size = 64;
    constexpr size_t tilingDataSize = sizeof(MultiAddRmsNormDynamicQuantTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t x1aByteSize = 64 * 2;
    constexpr size_t x1bByteSize = 64 * 2;
    constexpr size_t x1cByteSize = 64 * 2;
    constexpr size_t x1dByteSize = 64 * 2;
    constexpr size_t x1eByteSize = 64 * 2;
    constexpr size_t x2ByteSize = 64 * 2;
    constexpr size_t gammaByteSize = 64 * 2;
    constexpr size_t smoothScale1ByteSize = 64 * 2;
    constexpr size_t smoothScale2ByteSize = 64 * 2;
    constexpr size_t y1ByteSize = 64 * 1;
    constexpr size_t y2ByteSize = 64 * 1;
    constexpr size_t xByteSize = 64 * 2;
    constexpr size_t yByteSize = 64 * 2;
    constexpr size_t scale1ByteSize = 1 * 4;
    constexpr size_t scale2ByteSize = 1 * 4;
    std::vector<float> x1aHost(64, 1);
    std::vector<float> x1bHost(64, 1);
    std::vector<float> x1cHost(64, 1);
    std::vector<float> x1dHost(64, 1);
    std::vector<float> x1eHost(64, 1);
    std::vector<float> x2Host(64, 1);
    std::vector<float> gammaHost(64, 1);
    std::vector<float> smoothScale1Host(64, 1);
    std::vector<float> smoothScale2Host(64, 1);
    std::vector<int8_t> y1Host(64, 0);
    std::vector<int8_t> y2Host(64, 0);
    std::vector<float> xHost(64, 0);
    std::vector<float> yHost(64, 0);
    std::vector<float> scale1Host(1, 0);
    std::vector<float> scale2Host(1, 0);
    
    
    uint8_t* x1a = (uint8_t*)AscendC::GmAlloc(x1aByteSize);
    uint8_t* x1b = (uint8_t*)AscendC::GmAlloc(x1bByteSize);
    uint8_t* x1c = (uint8_t*)AscendC::GmAlloc(x1cByteSize);
    uint8_t* x1d = (uint8_t*)AscendC::GmAlloc(x1dByteSize);
    uint8_t* x1e = (uint8_t*)AscendC::GmAlloc(x1eByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* smoothScale1 = (uint8_t*)AscendC::GmAlloc(smoothScale1ByteSize);
    uint8_t* smoothScale2 = (uint8_t*)AscendC::GmAlloc(smoothScale2ByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1ByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2ByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale1 = (uint8_t*)AscendC::GmAlloc(scale1ByteSize);
    uint8_t* scale2 = (uint8_t*)AscendC::GmAlloc(scale2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(x1aHost[_i]); memcpy(x1a + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(x1bHost[_i]); memcpy(x1b + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(x1cHost[_i]); memcpy(x1c + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(x1dHost[_i]); memcpy(x1d + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(x1eHost[_i]); memcpy(x1e + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(x2Host[_i]); memcpy(x2 + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(gammaHost[_i]); memcpy(gamma + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(smoothScale1Host[_i]); memcpy(smoothScale1 + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 64; _i++) { uint16_t _h = FloatToHalf(smoothScale2Host[_i]); memcpy(smoothScale2 + _i * 2, &_h, 2); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/multi_add_rms_norm_dynamic_quant_tiling_data.h 中的字段定义
    //         参考 op_host/multi_add_rms_norm_dynamic_quant_tiling.cpp 中的 tiling 计算逻辑
    MultiAddRmsNormDynamicQuantTilingData* tilingData = reinterpret_cast<MultiAddRmsNormDynamicQuantTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/multi_add_rms_norm_dynamic_quant_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((multi_add_rms_norm_dynamic_quant<0>), numBlocks, x1a, x1b, x1c, x1d, x1e, x2, gamma, smoothScale1, smoothScale2, y1, y2, x, y, scale1, scale2, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(y1Host.data(), y1, y1ByteSize);
    { std::ofstream _ofs("int8_output_multi_add_rms_norm_dynamic_quant_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(y1Host.data()), y1ByteSize); }
    memcpy(y2Host.data(), y2, y2ByteSize);
    { std::ofstream _ofs("int8_output_multi_add_rms_norm_dynamic_quant_1.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(y2Host.data()), y2ByteSize); }
    memcpy(xHost.data(), x, xByteSize);
    { std::ofstream _ofs("float16_output_multi_add_rms_norm_dynamic_quant_2.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(xHost.data()), xByteSize); }
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float16_output_multi_add_rms_norm_dynamic_quant_3.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    memcpy(scale1Host.data(), scale1, scale1ByteSize);
    { std::ofstream _ofs("float32_output_multi_add_rms_norm_dynamic_quant_4.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(scale1Host.data()), scale1ByteSize); }
    memcpy(scale2Host.data(), scale2, scale2ByteSize);
    { std::ofstream _ofs("float32_output_multi_add_rms_norm_dynamic_quant_5.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(scale2Host.data()), scale2ByteSize); }
    
    AscendC::GmFree(x1a);
    AscendC::GmFree(x1b);
    AscendC::GmFree(x1c);
    AscendC::GmFree(x1d);
    AscendC::GmFree(x1e);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smoothScale1);
    AscendC::GmFree(smoothScale2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(scale1);
    AscendC::GmFree(scale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
