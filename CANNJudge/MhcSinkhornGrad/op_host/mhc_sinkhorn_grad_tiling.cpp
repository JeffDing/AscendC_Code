/*!
 * \file mhc_sinkhorn_grad_tiling.cpp
 * \brief MhcSinkhornGrad 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/mhc_sinkhorn_grad_tiling_data.h"
#include "../op_kernel/mhc_sinkhorn_grad_tiling_key.h"

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
//   blockFactor : [0:32)  = n * n（矩阵边长平方，kernel 侧开方得 n）
//                 [32:64) = T（矩阵个数，B*S 或 T）
//   ubFactor    : [0:32)  = T * n * n（总元素数，仅记录）
//                 [32:64) = numIters（迭代轮数）
//   totalNum    : 保留 = T * n * n（kernel 侧不读取该字段，仅作元素总数记录）
constexpr uint64_t MHG_LOW_MASK = 0xFFFFFFFFULL;

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

static ge::graphStatus MhcSinkhornGradTilingFunc(gert::TilingContext* context)
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

    MhcSinkhornGradTilingData* tiling = context->GetTilingData<MhcSinkhornGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    // ------------------------------------------------------------------
    // 1. 解析 gradOutput：(T, n, n) 或 (B, S, n, n)
    //    T = 除最后两维外所有维的乘积，n = 最后一维
    // ------------------------------------------------------------------
    const gert::StorageShape* gradShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShapePtr);
    const gert::Shape& gradShape = EnsureNotScalar(gradShapePtr->GetStorageShape());

    size_t dimNum = gradShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context, "gradOutput dimNum must be >= 2"), return ge::GRAPH_FAILED);

    int64_t n = gradShape.GetDim(dimNum - 1);
    if (n <= 0) { n = 1; }
    int64_t t = 1;
    for (size_t i = 0; i + 2 < dimNum; ++i) {
        int64_t d = gradShape.GetDim(i);
        if (d <= 0) { d = 1; }
        t *= d;
    }
    if (t <= 0) { t = 1; }

    // ------------------------------------------------------------------
    // 2. 由 normOut (2*numIters, T, n, n) 推导迭代轮数
    // ------------------------------------------------------------------
    int64_t numIters = 1;
    const gert::StorageShape* normShapePtr = context->GetInputShape(1);
    if (normShapePtr != nullptr) {
        const gert::Shape& normShape = normShapePtr->GetStorageShape();
        if (normShape.GetDimNum() >= 1 && normShape.GetDim(0) > 0) {
            numIters = normShape.GetDim(0) / 2;
        }
    }
    if (numIters <= 0) { numIters = 1; }

    int64_t totalNum = t * n * n;
    if (totalNum <= 0) { totalNum = 1; }

    // ------------------------------------------------------------------
    // 3. 多核切分：矩阵间完全独立（每步反传只依赖同矩阵数据），按矩阵连续均分
    // ------------------------------------------------------------------
    int64_t blockDim = coreNum;
    if (t < blockDim) { blockDim = t; }
    if (blockDim < 1) { blockDim = 1; }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // 4. UB 切分：kernel 侧使用编译期固定的面板元素数上限，恒不超过 UB 容量，
    //    无需 host 下发
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // 5. 打包 TilingData（字段语义与 kernel 注释保持一致）
    // ------------------------------------------------------------------
    tiling->totalNum = totalNum;
    tiling->blockFactor = static_cast<int64_t>(
        (static_cast<uint64_t>(n * n) & MHG_LOW_MASK) |
        ((static_cast<uint64_t>(t) & MHG_LOW_MASK) << 32));
    tiling->ubFactor = static_cast<int64_t>(
        (static_cast<uint64_t>(totalNum) & MHG_LOW_MASK) |
        ((static_cast<uint64_t>(numIters) & MHG_LOW_MASK) << 32));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(MHCSINKHORNGRAD_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(MHCSINKHORNGRAD_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMhcSinkhornGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MhcSinkhornGradCompileInfo {};

IMPL_OP_OPTILING(MhcSinkhornGrad).Tiling(MhcSinkhornGradTilingFunc).TilingParse<MhcSinkhornGradCompileInfo>(TilingParseForMhcSinkhornGrad);

} // namespace optiling
