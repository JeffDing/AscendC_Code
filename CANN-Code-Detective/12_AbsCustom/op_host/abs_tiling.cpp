/*!
 * \file abs_tiling.cpp
 * \brief Abs 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/abs_tiling_data.h"
#include "../op_kernel/abs_tiling_key.h"
#include <algorithm>

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

static ge::graphStatus AbsTilingFunc(gert::TilingContext* context)
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

    AbsTilingData* tiling = context->GetTilingData<AbsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    // 计算输入总元素数量
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& shape = inputShape->GetShape();
    int64_t totalNum = 1;
    int64_t dimNum = shape.GetDimNum();
    for (int64_t i = 0; i < dimNum; i++) {
        totalNum *= shape.GetDim(i);
    }

    // 多核切分: 沿元素维均分, 每核至少 1 个元素, 修正核数消除空转核
    // (kernel 侧自动按 min(blockFactor, ubFactor) 分轮, 核数不足时每核多轮)
    int64_t usedCore = (coreNum < totalNum) ? coreNum : totalNum;
    if (usedCore < 1) {
        usedCore = 1;
    }
    int64_t blockFactor = CeilDiv(totalNum, usedCore);
    usedCore = CeilDiv(totalNum, blockFactor);
    // 空张量时 CeilDiv(0, 0) 得 0, 需收敛回 1 避免 SetBlockDim(0)
    if (usedCore < 1) {
        usedCore = 1;
    }

    // 每轮 UB 处理元素数, 不超过每核元素数
    int64_t ubFactor = 1024;
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(ABS_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(ABS_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAbs([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AbsCompileInfo {};

IMPL_OP_OPTILING(Abs).Tiling(AbsTilingFunc).TilingParse<AbsCompileInfo>(TilingParseForAbs);

} // namespace optiling
