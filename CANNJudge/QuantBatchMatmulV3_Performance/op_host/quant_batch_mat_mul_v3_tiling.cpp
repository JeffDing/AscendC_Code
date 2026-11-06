/*!
 * \file quant_batch_mat_mul_v3_tiling.cpp
 * \brief QuantBatchMatMulV3 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_data.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_key.h"

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

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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

// TilingData 三个 int64 字段的位域打包约定（结构不可修改，host 与 kernel 必须严格一致）：
//   totalNum    : [0:32)   = K（累加维度大小）
//                 [32:64)  = M（x1 行数 / out 行数）
//   blockFactor : [0:32)   = mCnt（每核每次 UB 循环处理的 M 行数 = chunkM）
//                 [32:64)  = bf（每核负责的 x1 元素数 = mCnt * K）
//   ubFactor    : [0:32)   = chunkN（每次 UB 循环处理的 N 列数）
//                 [32:64)  = N（x2 列数 / out 列数）
// kernel 解码时对任一 <=0 的维度做 >=1 兜底，保证占位 tiling {32768,32768,32768}
// 不会崩溃（解码为 M=1, K=32768, N=1, mCnt=1 的安全单核单行场景）。
static constexpr uint64_t DIM_MASK = 0xFFFFFFFFULL;

static ge::graphStatus QuantBatchMatMulV3TilingFunc(gert::TilingContext* context)
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

    QuantBatchMatMulV3TilingData* tiling = context->GetTilingData<QuantBatchMatMulV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ------------------------------------------------------------------
    // 1. 解析 x1/x2 shape → M, K, N
    //    x1: (M, K) int8；x2: (K, N) int8；scale: (N) float；pertokenScale: (M) float；
    //    out: (M, N) bf16。本算子无 transpose attrs。
    // ------------------------------------------------------------------
    const gert::StorageShape* x1ShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1ShapePtr);
    const gert::Shape& x1Shape = EnsureNotScalar(x1ShapePtr->GetStorageShape());

    const gert::StorageShape* x2ShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2ShapePtr);
    const gert::Shape& x2Shape = EnsureNotScalar(x2ShapePtr->GetStorageShape());

    int64_t M = 1, K = 1, N = 1;
    if (x1Shape.GetDimNum() >= 2) {
        M = x1Shape.GetDim(0);
        K = x1Shape.GetDim(1);
    } else if (x1Shape.GetDimNum() == 1) {
        K = x1Shape.GetDim(0);
    }
    if (x2Shape.GetDimNum() >= 2) {
        N = x2Shape.GetDim(1);
        if (x2Shape.GetDim(0) != K) {
            OP_LOGE(context, "x2 first dim %ld does not match x1 second dim %ld", x2Shape.GetDim(0), K);
            return ge::GRAPH_FAILED;
        }
    } else if (x2Shape.GetDimNum() == 1) {
        N = x2Shape.GetDim(0);
    }
    if (M <= 0) { M = 1; }
    if (K <= 0) { K = 1; }
    if (N <= 0) { N = 1; }

    // ------------------------------------------------------------------
    // 2. UB 切分策略（kernel 侧 AllocBuffers 精确核算，chunkM=16 / chunkN=192 / kBlk=32）：
    //    x1   = mAl * kAl * 4B = 16*32*4 = 2KB   ×2（int8 中转 + float 视图）
    //    x2   = kAl * stride * 4B = 32*192*4 = 24KB ×2（int8 中转 + float 视图）
    //    half x1/x2 两段 Cast 的 half 中转(2B) 见 kernel AllocBuffers
    //    acc/tmp/accF32 = mAl*stride*4B = 16*192*4 = 12KB ×3
    //    outBF16 = mAl*stride*2B = 6KB
    //    合计 ≈ 110KB < 192KB（UB 安全预算），详见 README/报告说明。
    // ------------------------------------------------------------------
    int64_t chunkM = M;
    if (chunkM > 16) { chunkM = 16; }
    if (chunkM < 1) { chunkM = 1; }

    int64_t chunkN = N;
    if (chunkN > 192) { chunkN = 192; }
    if (chunkN < 1) { chunkN = 1; }

    int64_t totalMChunks = CeilDiv(M, chunkM);
    int64_t blockDim = totalMChunks;
    if (blockDim > coreNum) { blockDim = coreNum; }
    if (blockDim < 1) { blockDim = 1; }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    uint64_t totalPacked = (static_cast<uint64_t>(K) & DIM_MASK) |
        ((static_cast<uint64_t>(M) & DIM_MASK) << 32);
    uint64_t blockPacked = (static_cast<uint64_t>(chunkM) & DIM_MASK) |
        (((static_cast<uint64_t>(chunkM) * K) & DIM_MASK) << 32);
    uint64_t ubPacked = (static_cast<uint64_t>(chunkN) & DIM_MASK) |
        ((static_cast<uint64_t>(N) & DIM_MASK) << 32);

    tiling->totalNum = static_cast<int64_t>(totalPacked);
    tiling->blockFactor = static_cast<int64_t>(blockPacked);
    tiling->ubFactor = static_cast<int64_t>(ubPacked);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantBatchMatMulV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantBatchMatMulV3CompileInfo {};

IMPL_OP_OPTILING(QuantBatchMatMulV3).Tiling(QuantBatchMatMulV3TilingFunc).TilingParse<QuantBatchMatMulV3CompileInfo>(TilingParseForQuantBatchMatMulV3);

} // namespace optiling
