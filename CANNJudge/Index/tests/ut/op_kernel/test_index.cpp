/*!
 * \file test_index.cpp
 * \brief Index 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "index_tiling.h"
#include "../../../op_kernel/index.cpp"

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

class IndexKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "IndexKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "IndexKernelTest TearDown" << endl;
    }
};

TEST_F(IndexKernelTest, test_kernel_run)
{
    constexpr size_t size = 1048576;
    constexpr size_t tilingDataSize = sizeof(IndexTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 1048576 * 4;
    constexpr size_t indexed_sizesByteSize = 2 * 8;
    constexpr size_t indicesByteSize = 10 * 8;
    constexpr size_t yByteSize = 1280 * 4;
    std::vector<float> xHost(1048576, 1);
    std::vector<int64_t> indexed_sizesHost(2, 1);
    std::vector<int64_t> indicesHost(10, 1);
    std::vector<float> yHost(1280, 0);
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* indexed_sizes = (uint8_t*)AscendC::GmAlloc(indexed_sizesByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x, xHost.data(), xByteSize);
    memcpy(indexed_sizes, indexed_sizesHost.data(), indexed_sizesByteSize);
    memcpy(indices, indicesHost.data(), indicesByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/index_tiling_data.h 中的字段定义
    //         参考 op_host/index_tiling.cpp 中的 tiling 计算逻辑
    IndexTilingData* tilingData = reinterpret_cast<IndexTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/index_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((index<1>), numBlocks, x, indexed_sizes, indices, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float32_output_index_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(indexed_sizes);
    AscendC::GmFree(indices);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
