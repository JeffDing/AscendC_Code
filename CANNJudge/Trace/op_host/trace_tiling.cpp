/*!
 * \file trace_tiling.cpp
 * \brief Trace 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/trace_tiling_data.h"
#include "../op_kernel/trace_tiling_key.h"

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

static ge::graphStatus TraceTilingFunc(gert::TilingContext* context)
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

    TraceTilingData* tiling = context->GetTilingData<TraceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 解析输入 (M, N), 退化场景按 1 处理
    const auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();
    int64_t m = 1;
    int64_t n = 1;
    if (inputShape.GetDimNum() >= 2) {
        m = inputShape.GetDim(0);
        n = inputShape.GetDim(1);
    } else if (inputShape.GetDimNum() == 1) {
        m = inputShape.GetDim(0);
    }
    if (m < 1) {
        m = 1;
    }
    if (n < 1) {
        n = 1;
    }
    const int64_t diagNum = (m < n) ? m : n;

    // dtype: fp16/bf16 走 half kernel, 其余走 float kernel
    bool isHalf = false;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
                                 inputDesc->GetDataType() == ge::DT_BF16)) {
        isHalf = true;
    }

    // 多核切分: 每核至少 1 个对角元, 修正核数消除空转核
    // (kernel 端自动按 min(blockFactor, UB 安全上限) 分轮, 核数不足时每核多轮)
    // fp16 固定单核: 核内 fp32 精确累加后单次 half 舍入直写, 精度最优且不依赖
    // workspace 汇总机制(half 多核原子累加存在超精度阈值风险, workspace 指针
    // 在真机 aclnn 调用路径下不可靠, 实测会导致 kernel 挂起超时)
    int64_t usedCore = (coreNum < diagNum) ? coreNum : diagNum;
    if (isHalf) {
        usedCore = 1;
    }
    if (usedCore < 1) {
        usedCore = 1;
    }
    int64_t blockFactor = CeilDiv(diagNum, usedCore);
    usedCore = CeilDiv(diagNum, blockFactor);

    // TODO: 设置 tiling 数据
    // ubFactor 字段承载矩阵列数 N(kernel 计算对角线 stride N+1 的必需信息)
    tiling->totalNum = diagNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = n;

    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalf) {
        tilingKey = GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTrace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TraceCompileInfo {};

IMPL_OP_OPTILING(Trace).Tiling(TraceTilingFunc).TilingParse<TraceCompileInfo>(TilingParseForTrace);

} // namespace optiling
