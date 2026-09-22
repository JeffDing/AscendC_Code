/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t UB_ALIGN_BYTES = 32;         // DataCopy 32B 对齐粒度
constexpr int64_t UB_TOTAL_BUFFER_NUM = 4;     // UB 内 buffer 总份数：输入/输出各双缓冲

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
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

    // 输入张量总元素数
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalNum *= xShape.GetDim(i);
    }

    // 输入 dtype 对应的元素字节数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t typeSize = ge::GetSizeByDataType(inputDesc->GetDataType());
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "unsupported data type"), return ge::GRAPH_FAILED);

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 多核均分：每个核处理的元素数（向上取整，尾核不足时在 kernel 内截断）
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    if (blockFactor <= 0 || (blockFactor < MIN_SPLIT_THRESHOLD && blockFactor < totalNum)) {
        // 数据量过小时不拆多核，保持整块处理；空输入退化为单块
        blockFactor = (totalNum > 0) ? totalNum : 1;
    }

    // 单次 UB 循环元素数：输入/输出各双缓冲，共占 UB_TOTAL_BUFFER_NUM * ubFactor * typeSize
    int64_t maxUbElems = (static_cast<int64_t>(ubSize) / UB_TOTAL_BUFFER_NUM) / typeSize;
    int64_t alignElems = UB_ALIGN_BYTES / typeSize;
    if (maxUbElems > alignElems) {
        maxUbElems = FloorAlign(maxUbElems, alignElems);
    }
    int64_t ubFactor = (maxUbElems < blockFactor) ? maxUbElems : blockFactor;
    if (ubFactor <= 0) {
        ubFactor = 1;
    }

    // 实际启动核数：数据量不足时不需要全部核参与
    int64_t blockDim = CeilDiv(totalNum, blockFactor);
    if (blockDim < 1) {
        blockDim = 1;
    }

    // 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto tilingInputDesc = context->GetInputDesc(0);
    if (tilingInputDesc != nullptr && (tilingInputDesc->GetDataType() == ge::DT_FLOAT16 || tilingInputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
