/*!
 * \file bn_training_update_tiling.cpp
 * \brief BnTrainingUpdate 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/bn_training_update_tiling_data.h"
#include "../op_kernel/bn_training_update_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// TilingData 打包位域掩码（与 kernel 侧保持一致）
constexpr uint64_t HW_MASK = 0x1FFFFFULL;      // H * W，占 21 bit
constexpr uint64_t C_MASK = 0x1FFFULL;         // 通道数，占 13 bit
constexpr uint64_t ROWS_MASK = 0x1FFFFFFULL;   // N * C，占 25 bit

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

static ge::graphStatus BnTrainingUpdateTilingFunc(gert::TilingContext* context)
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

    BnTrainingUpdateTilingData* tiling = context->GetTilingData<BnTrainingUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    // ------------------------------------------------------------------
    // 1. 解析 x 的 (N, C, H, W)，兼容 1~4 维退化输入
    // ------------------------------------------------------------------
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();

    int64_t n = 1;
    int64_t c = 1;
    int64_t h = 1;
    int64_t w = 1;
    size_t dimNum = xShape.GetDimNum();
    if (dimNum >= 4) {
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
    if (h <= 0) { h = 1; }
    if (w <= 0) { w = 1; }

    // 用 sum 的 shape 校正通道数（sum/square_sum/scale/offset 均为 (C,)）
    const gert::StorageShape* sumShapePtr = context->GetInputShape(1);
    if (sumShapePtr != nullptr) {
        const gert::Shape& sumShape = sumShapePtr->GetStorageShape();
        if (sumShape.GetDimNum() >= 1 && sumShape.GetDim(0) > 0) {
            c = sumShape.GetDim(0);
        }
    }
    if (c <= 0) { c = 1; }

    int64_t hw = h * w;                  // 每行（同一 batch、同一通道）元素数
    int64_t totalNum = n * c * hw;       // 元素总数
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

    // 输入 dtype 决定字节宽度
    auto inputDesc = context->GetInputDesc(0);
    bool isHalf = (inputDesc != nullptr) &&
                  (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16);

    // ------------------------------------------------------------------
    // 2. 多核切分：按元素总量均分，每个核处理一段连续元素
    // ------------------------------------------------------------------
    int64_t blockDim = coreNum;
    if (totalNum < blockDim) {
        blockDim = totalNum;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // 3. UB 切分：kernel 侧使用编译期固定的单轮元素数（UB_ELEMS_FIXED），
    //    恒不超过 UB 容量，无需 host 下发
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // 4. 打包 TilingData（结构不可改，字段语义与 kernel 注释保持一致）
    //    totalNum    : 保留（元素总数 N*C*H*W，kernel 由 rows*hw 推导）
    //    blockFactor : [0:21) = hw = H*W，[21:34) = C，[34:59) = rows = N*C
    //    ubFactor    : [0:32) = epsilon 的 IEEE-754 位模式
    // ------------------------------------------------------------------
    union { uint32_t u; float f; } epsConv;
    epsConv.f = epsilon;
    uint32_t epsBits = epsConv.u;

    tiling->totalNum = totalNum;
    tiling->blockFactor = static_cast<int64_t>(
        (static_cast<uint64_t>(hw) & HW_MASK) | ((static_cast<uint64_t>(c) & C_MASK) << 21) |
        ((static_cast<uint64_t>(n * c) & ROWS_MASK) << 34));
    tiling->ubFactor = static_cast<int64_t>(static_cast<uint64_t>(epsBits));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalf) {
        tilingKey = GET_TPL_TILING_KEY(BNTRAININGUPDATE_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(BNTRAININGUPDATE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBnTrainingUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BnTrainingUpdateCompileInfo {};

IMPL_OP_OPTILING(BnTrainingUpdate).Tiling(BnTrainingUpdateTilingFunc).TilingParse<BnTrainingUpdateCompileInfo>(TilingParseForBnTrainingUpdate);

} // namespace optiling