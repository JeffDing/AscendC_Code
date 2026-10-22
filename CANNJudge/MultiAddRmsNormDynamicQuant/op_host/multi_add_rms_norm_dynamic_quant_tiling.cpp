/*!
 * \file multi_add_rms_norm_dynamic_quant_tiling.cpp
 * \brief MultiAddRmsNormDynamicQuant 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/multi_add_rms_norm_dynamic_quant_tiling_data.h"
#include "../op_kernel/multi_add_rms_norm_dynamic_quant_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus MultiAddRmsNormDynamicQuantTilingFunc(gert::TilingContext* context)
{
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

    MultiAddRmsNormDynamicQuantTilingData* tiling = context->GetTilingData<MultiAddRmsNormDynamicQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 输入 x1a: (B, H)，固定 2D ND
    const gert::StorageShape* xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const gert::Shape& xShape = xStorage->GetShape();
    OP_CHECK_IF(
        xShape.GetDimNum() != 2,
        OP_LOGE(context, "input x1a dim num %zu != 2", xShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    const int64_t bSize = xShape.GetDim(0);
    const int64_t hSize = xShape.GetDim(1);
    OP_CHECK_IF(
        bSize <= 0 || hSize <= 0,
        OP_LOGE(context, "invalid shape B=%ld H=%ld", bSize, hSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        bSize > 8192 || hSize > 16384,
        OP_LOGE(context, "shape out of range B=%ld H=%ld", bSize, hSize),
        return ge::GRAPH_FAILED);

    // epsilon：OPTIONAL Float，缺省 1e-6
    float epsVal = 1e-6f;
    const auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        const float* epsPtr = attrs->GetFloat(0);
        if (epsPtr != nullptr) {
            epsVal = *epsPtr;
        }
    }
    if (!(epsVal > 0.0f)) {
        epsVal = 0.0f;
    }

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const bool isBf16 = (inputDesc->GetDataType() == ge::DT_BF16);
    if (inputDesc->GetDataType() == ge::DT_FLOAT16 || isBf16) {
        tilingKey = GET_TPL_TILING_KEY(MULTIADDRMSNORMDYNAMICQUANT_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(MULTIADDRMSNORMDYNAMICQUANT_TPL_SCH_MODE_1);
    }

    // TilingData 编码（kernel 解码，详见开发文档）：
    //   ubFactor  = H | (isBf16 << 15)   （H <= 16384 不占用 bit15）
    //   totalNum  = B * H
    //   blockFactor = epsilon 的 float32 位模式（低 32 位，kernel 端防御性还原）
    tiling->totalNum = bSize * hSize;
    tiling->ubFactor = hSize | (isBf16 ? (static_cast<int64_t>(1) << 15) : static_cast<int64_t>(0));
    union EpsConv {
        float f;
        uint32_t u;
    };
    EpsConv epsConv;
    epsConv.f = epsVal;
    tiling->blockFactor = static_cast<int64_t>(static_cast<uint64_t>(epsConv.u));

    // 核间按行切分，每核至少一行
    context->SetBlockDim(static_cast<uint32_t>((bSize < coreNum) ? bSize : coreNum));

    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMultiAddRmsNormDynamicQuant([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MultiAddRmsNormDynamicQuantCompileInfo {};

IMPL_OP_OPTILING(MultiAddRmsNormDynamicQuant).Tiling(MultiAddRmsNormDynamicQuantTilingFunc).TilingParse<MultiAddRmsNormDynamicQuantCompileInfo>(TilingParseForMultiAddRmsNormDynamicQuant);

} // namespace optiling
