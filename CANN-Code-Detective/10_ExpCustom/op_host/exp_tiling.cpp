/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t HALF_TYPE_SIZE = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t QUEUE_NUM = 2;
constexpr int64_t BUFFER_NUM_IN_UB = 2;
// UB 池按 3/4 比例使用，预留系统开销（CPU 仿真池上限亦为 0.75 * UB_SIZE）
constexpr int64_t UB_USAGE_NUMERATOR = 3;
constexpr int64_t UB_USAGE_DENOMINATOR = 4;

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

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
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

    const gert::StorageShape* storageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape inShape = EnsureNotScalar(storageShape->GetStorageShape());
    const int64_t totalNum = inShape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "totalNum is negative"), return ge::GRAPH_FAILED);

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const bool isHalfType = (dtype == ge::DT_FLOAT16) || (dtype == ge::DT_BF16);
    if (isHalfType) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    }

    const int64_t typeSize = isHalfType ? HALF_TYPE_SIZE : TYPE_SIZE;
    // 32B 搬运对齐粒度对应的元素个数
    const int64_t alignNum = static_cast<int64_t>(GetUbBlockSize(context)) / typeSize;

    // 核间切分：小规模数据不拆核，避免任务过碎；
    // blockFactor 按 32B 粒度向上对齐，保证多核场景各核 GM 起始偏移满足搬运对齐要求
    int64_t blockDim = 1;
    if (totalNum >= MIN_SPLIT_THRESHOLD) {
        blockDim = std::min(coreNum, totalNum);
    }
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum);
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }
    blockDim = std::max<int64_t>(1, CeilDiv(totalNum, blockFactor));

    // 核内切分：输入/输出两条队列，每条队列 BUFFER_NUM_IN_UB 个 buffer，且按 32 字节块对齐
    const int64_t usableUbSize = static_cast<int64_t>(ubSize) * UB_USAGE_NUMERATOR / UB_USAGE_DENOMINATOR;
    const int64_t ubTotalNum = FloorDiv(usableUbSize, typeSize * QUEUE_NUM * BUFFER_NUM_IN_UB);
    const int64_t ubFactorCap = std::max(alignNum, FloorAlign(ubTotalNum, alignNum));
    const int64_t ubFactor = std::min(blockFactor, ubFactorCap);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
