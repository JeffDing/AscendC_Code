/*!
 * \file index_tiling.cpp
 * \brief Index 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/index_tiling_data.h"
#include "../op_kernel/index_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// TilingData 三个 int64 字段按位打包（结构文件不可改）
// blockFactor: [0:4)=ndim  [4:24)=dim0  [24:44)=dim1  [44:64)=dim2
// ubFactor   : [0:20)=dim3  [20:40)=dim4  [40:62)=Q（indices 元素总数）  [62:64)=dtypeCode（0=2B,1=4B,2=8B）
// totalNum   : [0:20)=dim5  [20:40)=dim6  [40:60)=dim7（ndim>5 时使用）
constexpr int64_t MAX_NDIM = 8;
constexpr uint64_t DIM_MASK_U = (1ULL << 20) - 1ULL;
constexpr uint64_t Q_MASK_U = (1ULL << 22) - 1ULL;
constexpr uint64_t DTYPE_CODE_2B = 0ULL;
constexpr uint64_t DTYPE_CODE_4B = 1ULL;
constexpr uint64_t DTYPE_CODE_8B = 2ULL;

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

static ge::graphStatus IndexTilingFunc(gert::TilingContext* context)
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

    IndexTilingData* tiling = context->GetTilingData<IndexTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    const gert::StorageShape* xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const gert::StorageShape* indicesStorage = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesStorage);
    const gert::Shape& xShape = xStorage->GetStorageShape();
    const gert::Shape& indicesShape = indicesStorage->GetStorageShape();

    int64_t ndim = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(
        ndim < 1 || ndim > MAX_NDIM,
        OP_LOGE(context, "index: unsupported ndim %ld (1..%ld)", ndim, MAX_NDIM),
        return ge::GRAPH_FAILED);

    uint64_t dims[MAX_NDIM] = {1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL};
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = xShape.GetDim(i);
        OP_CHECK_IF(
            d < 0 || static_cast<uint64_t>(d) > DIM_MASK_U,
            OP_LOGE(context, "index: x dim[%ld]=%ld out of range", i, d),
            return ge::GRAPH_FAILED);
        dims[i] = static_cast<uint64_t>(d);
    }

    uint64_t q = 1ULL;
    for (size_t i = 0; i < indicesShape.GetDimNum(); ++i) {
        int64_t d = indicesShape.GetDim(i);
        if (d <= 0) {
            q = 0ULL;
            break;
        }
        q *= static_cast<uint64_t>(d);
    }
    OP_CHECK_IF(q > Q_MASK_U, OP_LOGE(context, "index: indices num %lu too large", q), return ge::GRAPH_FAILED);

    // 根据输入 dtype 选择 tilingKey 与元素字节数编码（int64 复用 float 模板，按字节数承载）
    uint64_t tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_1);
    uint64_t dtypeCode = DTYPE_CODE_4B;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr) {
        ge::DataType dt = inputDesc->GetDataType();
        if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
            tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_0);
            dtypeCode = DTYPE_CODE_2B;
        } else if (dt == ge::DT_INT64 || dt == ge::DT_DOUBLE) {
            tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_1);
            dtypeCode = DTYPE_CODE_8B;
        } else {
            tilingKey = GET_TPL_TILING_KEY(INDEX_TPL_SCH_MODE_1);
            dtypeCode = DTYPE_CODE_4B;
        }
    }

    uint64_t blockFactor = 0ULL;
    blockFactor |= static_cast<uint64_t>(ndim) & 0xFULL;
    blockFactor |= (dims[0] & DIM_MASK_U) << 4;
    blockFactor |= (dims[1] & DIM_MASK_U) << 24;
    blockFactor |= (dims[2] & DIM_MASK_U) << 44;

    uint64_t ubFactor = 0ULL;
    ubFactor |= dims[3] & DIM_MASK_U;
    ubFactor |= (dims[4] & DIM_MASK_U) << 20;
    ubFactor |= (q & Q_MASK_U) << 40;
    ubFactor |= (dtypeCode & 0x3ULL) << 62;

    uint64_t totalNum = 0ULL;
    totalNum |= dims[5] & DIM_MASK_U;
    totalNum |= (dims[6] & DIM_MASK_U) << 20;
    totalNum |= (dims[7] & DIM_MASK_U) << 40;

    tiling->totalNum = static_cast<int64_t>(totalNum);
    tiling->blockFactor = static_cast<int64_t>(blockFactor);
    tiling->ubFactor = static_cast<int64_t>(ubFactor);

    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIndex([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct IndexCompileInfo {};

IMPL_OP_OPTILING(Index).Tiling(IndexTilingFunc).TilingParse<IndexCompileInfo>(TilingParseForIndex);

} // namespace optiling