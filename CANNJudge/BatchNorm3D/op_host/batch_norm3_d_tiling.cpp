/*!
 * \file batch_norm3_d_tiling.cpp
 * \brief BatchNorm3D 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/batch_norm3_d_tiling_data.h"
#include "../op_kernel/batch_norm3_d_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// TilingData 打包位域约定（结构不可改，host 与 kernel 两侧必须严格一致）：
//   blockFactor : [0:39)  = DHW = D * H * W（每个 (n, c) 段的元素数）
//                 [39:52) = C（通道数，13 bit，最大 8191）
//   ubFactor    : [0:32)  = epsilon 的 IEEE-754 位模式
//                 [32:64) = N（batch 数，即每通道的段数）
//   totalNum    : 保留 = N * C * DHW（kernel 侧不读取该字段，仅作元素总数记录）
constexpr uint64_t DHW_MASK = 0x7FFFFFFFFFULL;  // 39 bit
constexpr uint64_t C_MASK = 0x1FFFULL;          // 13 bit
constexpr uint64_t N_MASK = 0xFFFFFFFFULL;      // 32 bit

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

static ge::graphStatus BatchNorm3DTilingFunc(gert::TilingContext* context)
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

    BatchNorm3DTilingData* tiling = context->GetTilingData<BatchNorm3DTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    // ------------------------------------------------------------------
    // 1. 解析 x 的 (N, C, D, H, W)，兼容更低维度的退化输入
    // ------------------------------------------------------------------
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();

    int64_t n = 1;
    int64_t c = 1;
    int64_t d = 1;
    int64_t h = 1;
    int64_t w = 1;
    size_t dimNum = xShape.GetDimNum();
    if (dimNum >= 5) {
        n = xShape.GetDim(0);
        c = xShape.GetDim(1);
        d = xShape.GetDim(2);
        h = xShape.GetDim(3);
        w = xShape.GetDim(4);
    } else if (dimNum == 4) {
        n = xShape.GetDim(0);
        c = xShape.GetDim(1);
        h = xShape.GetDim(2);
        w = xShape.GetDim(3);
    } else if (dimNum == 3) {
        c = xShape.GetDim(0);
        h = xShape.GetDim(1);
        w = xShape.GetDim(2);
    } else if (dimNum == 2) {
        c = xShape.GetDim(0);
        h = xShape.GetDim(1);
    } else if (dimNum == 1) {
        h = xShape.GetDim(0);
    }
    if (n <= 0) { n = 1; }
    if (d <= 0) { d = 1; }
    if (h <= 0) { h = 1; }
    if (w <= 0) { w = 1; }

    // 用 scale 的 shape 校正通道数（scale / offset 均为 (C,)）
    const gert::StorageShape* scaleShapePtr = context->GetInputShape(1);
    if (scaleShapePtr != nullptr) {
        const gert::Shape& scaleShape = scaleShapePtr->GetStorageShape();
        if (scaleShape.GetDimNum() >= 1 && scaleShape.GetDim(0) > 0) {
            c = scaleShape.GetDim(0);
        }
    }
    if (c <= 0) { c = 1; }

    int64_t dhw = d * h * w;              // 每个 (n, c) 段的元素数（同一通道归约集合的一部分）
    int64_t totalNum = n * c * dhw;       // 元素总数
    if (totalNum <= 0) { totalNum = 1; }

    // epsilon 属性
    float epsilon = 1e-5f;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* epsPtr = attrs->GetAttrPointer<float>(0);
        if (epsPtr != nullptr) {
            epsilon = *epsPtr;
        }
    }

    // 输入 dtype 决定走 half 还是 float kernel
    auto inputDesc = context->GetInputDesc(0);
    bool isHalf = (inputDesc != nullptr) &&
                  (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16);

    // ------------------------------------------------------------------
    // 2. 多核切分：按通道均分，每个核负责一段连续通道 [channelStart, channelEnd)
    //    （通道内的统计归约互相独立，天然无跨核依赖）
    // ------------------------------------------------------------------
    int64_t blockDim = coreNum;
    if (c < blockDim) {
        blockDim = c;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // 3. UB 切分：kernel 侧使用编译期固定的单块元素数，恒不超过 UB 容量，
    //    无需 host 下发
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // 4. 打包 TilingData（结构不可改，字段语义与 kernel 注释保持一致）
    // ------------------------------------------------------------------
    union { uint32_t u; float f; } epsConv;
    epsConv.f = epsilon;
    uint32_t epsBits = epsConv.u;

    tiling->totalNum = totalNum;
    tiling->blockFactor = static_cast<int64_t>(
        (static_cast<uint64_t>(dhw) & DHW_MASK) |
        ((static_cast<uint64_t>(c) & C_MASK) << 39));
    tiling->ubFactor = static_cast<int64_t>(
        static_cast<uint64_t>(epsBits) |
        ((static_cast<uint64_t>(n) & N_MASK) << 32));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalf) {
        tilingKey = GET_TPL_TILING_KEY(BATCHNORM3D_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(BATCHNORM3D_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBatchNorm3D([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BatchNorm3DCompileInfo {};

IMPL_OP_OPTILING(BatchNorm3D).Tiling(BatchNorm3DTilingFunc).TilingParse<BatchNorm3DCompileInfo>(TilingParseForBatchNorm3D);

} // namespace optiling