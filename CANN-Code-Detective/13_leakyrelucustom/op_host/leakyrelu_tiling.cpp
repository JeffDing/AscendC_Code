/*!
 * \file leakyrelu_tiling.cpp
 * \brief Leakyrelu 算子 Tiling 实现
 */

#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/leakyrelu_tiling_data.h"
#include "../op_kernel/leakyrelu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// negative_slope 的 PyTorch 默认值，属性缺失或非法时回退
constexpr float LEAKY_RELU_DEFAULT_SLOPE = 0.01f;
// slope 位模式为 0 时作为"未设置"哨兵，kernel 侧回退默认值（host 对 slope==0 编码为 denormal 位 1）
constexpr uint32_t LEAKY_RELU_SLOPE_SENTINEL = 0U;
constexpr uint32_t LEAKY_RELU_SLOPE_ZERO_CODE = 1U;
// 每个元素在 UB 中的字节预算：输入/输出各双缓冲(2*sizeof(T) + 2*sizeof(T))，
// half 额外两个 float32 中间缓冲(2*sizeof(float))，两种 dtype 均为 16 字节/元素
constexpr int64_t UB_BYTES_PER_ELEM = 16;
constexpr uint64_t LEAKY_RELU_UB_DIVISOR = 2;   // 只使用一半 UB，预留框架开销

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 用位模式判断 float 是否有限，避免 host 侧依赖 <cmath> 的设备端差异
static inline bool IsFiniteFloatBits(uint32_t bits)
{
    return (bits & 0x7F800000U) != 0x7F800000U;
}

static ge::graphStatus LeakyreluTilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    LeakyreluTilingData* tiling = context->GetTilingData<LeakyreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    // 1) 读取 negative_slope 属性（IR 原型中 index 0），缺失/非法时回退默认值 0.01
    float negativeSlope = LEAKY_RELU_DEFAULT_SLOPE;
    const float* slopePtr = context->GetAttrs()->GetAttrPointer<float>(0);
    if (slopePtr != nullptr) {
        uint32_t slopeBits = 0;
        (void)memcpy(&slopeBits, slopePtr, sizeof(float));
        if (IsFiniteFloatBits(slopeBits)) {
            negativeSlope = *slopePtr;
        }
    }
    uint32_t slopeCode = LEAKY_RELU_SLOPE_SENTINEL;
    if (negativeSlope != 0.0f) {
        (void)memcpy(&slopeCode, &negativeSlope, sizeof(float));
    } else {
        // slope == 0 时编码为最小 denormal 位模式，数值上等价 0，
        // 从而把位模式 0 留作"未设置"哨兵（kernel 侧回退默认 0.01）
        slopeCode = LEAKY_RELU_SLOPE_ZERO_CODE;
    }

    // 2) 总元素数（StorageShape 的 runtime shape 已按 ND 连续展开）
    int64_t totalNum = 0;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape != nullptr) {
        const gert::Shape& storageShape = xShape->GetStorageShape();
        if (storageShape.GetDimNum() != 0) {
            totalNum = storageShape.GetShapeSize();
        }
    }
    if (totalNum < 1) {
        // 标量/空 shape 兜底为 1 个元素，保证 kernel 循环至少处理一次
        totalNum = 1;
    }

    // 3) 多核切分：每个核至少 MIN_SPLIT_THRESHOLD 个元素，小张量少开核
    int64_t blockDim = coreNum;
    const int64_t needCores = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (blockDim > needCores) {
        blockDim = needCores;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    const int64_t blockFactor = CeilDiv(totalNum, blockDim);

    // 4) UB 分块大小：16 字节/元素预算，对齐 32B（DMA 粒度），双缓冲输入输出
    int64_t elemBytes = TYPE_SIZE;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        elemBytes = 2;
    }
    int64_t ubElems = static_cast<int64_t>(ubSize) / LEAKY_RELU_UB_DIVISOR / UB_BYTES_PER_ELEM;
    if (ubElems < 1) {
        ubElems = 1;
    }
    ubElems = FloorAlign(ubElems, 32 / elemBytes);
    if (ubElems < 1) {
        ubElems = 1;
    }

    // ubFactor: bit[0:32) = 每轮 UB 元素数, bit[32:64) = negative_slope 的 IEEE-754 位模式
    // totalNum/blockFactor 直接存放标量，不按位域拆分，规避退化 tiling 解码歧义
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = static_cast<int64_t>(static_cast<uint64_t>(static_cast<uint32_t>(ubElems)) |
                                            (static_cast<uint64_t>(slopeCode) << 32));

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(LEAKYRELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLeakyrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LeakyreluCompileInfo {};

IMPL_OP_OPTILING(Leakyrelu).Tiling(LeakyreluTilingFunc).TilingParse<LeakyreluCompileInfo>(TilingParseForLeakyrelu);

} // namespace optiling
