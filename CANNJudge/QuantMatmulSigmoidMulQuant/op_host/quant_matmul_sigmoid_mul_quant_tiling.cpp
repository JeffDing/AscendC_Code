/*!
 * \file quant_matmul_sigmoid_mul_quant_tiling.cpp
 * \brief QuantMatmulSigmoidMulQuant 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/quant_matmul_sigmoid_mul_quant_tiling_data.h"
#include "../op_kernel/quant_matmul_sigmoid_mul_quant_tiling_key.h"

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
//   blockFactor : [0:24)  = M（输出行数）
//                 [24:48) = K（累加维度）
//                 [48]    = transposeX1
//                 [49]    = transposeX2
//                 [50:64) = reserved
//   ubFactor    : [0:24)  = N（输出列数）
//                 [24:36) = CHUNK_M（每次 UB 处理的 M 行数）
//                 [36:48) = CHUNK_N（每次 UB 处理的 N 列数）
//                 [48:64) = reserved
//   totalNum    : M * N（元素总数）
constexpr uint64_t M_MASK = 0xFFFFFFULL;       // 24 bit
constexpr uint64_t K_MASK = 0xFFFFFFULL;       // 24 bit
constexpr uint64_t N_MASK = 0xFFFFFFULL;       // 24 bit
constexpr uint64_t CHUNK_M_MASK = 0xFFFULL;    // 12 bit
constexpr uint64_t CHUNK_N_MASK = 0xFFFULL;    // 12 bit

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

static ge::graphStatus QuantMatmulSigmoidMulQuantTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    (void)coreNum;

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    QuantMatmulSigmoidMulQuantTilingData* tiling = context->GetTilingData<QuantMatmulSigmoidMulQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ------------------------------------------------------------------
    // 1. 解析 x1/x2 shape + attrs → M, K, N
    // ------------------------------------------------------------------
    const gert::StorageShape* x1ShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1ShapePtr);
    const gert::Shape& x1Shape = EnsureNotScalar(x1ShapePtr->GetStorageShape());

    const gert::StorageShape* x2ShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2ShapePtr);
    const gert::Shape& x2Shape = EnsureNotScalar(x2ShapePtr->GetStorageShape());

    bool transposeX1 = false;
    bool transposeX2 = false;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* t1Ptr = attrs->GetAttrPointer<bool>(0);
        if (t1Ptr != nullptr) { transposeX1 = *t1Ptr; }
        const bool* t2Ptr = attrs->GetAttrPointer<bool>(1);
        if (t2Ptr != nullptr) { transposeX2 = *t2Ptr; }
    }

    int64_t M = 1, K = 1, N = 1;
    if (x1Shape.GetDimNum() >= 2) {
        if (transposeX1) {
            K = x1Shape.GetDim(0);
            M = x1Shape.GetDim(1);
        } else {
            M = x1Shape.GetDim(0);
            K = x1Shape.GetDim(1);
        }
    }
    if (x2Shape.GetDimNum() >= 2) {
        if (transposeX2) {
            N = x2Shape.GetDim(0);
        } else {
            N = x2Shape.GetDim(1);
        }
    }
    if (M <= 0) { M = 1; }
    if (K <= 0) { K = 1; }
    if (N <= 0) { N = 1; }

    // ------------------------------------------------------------------
    // 2. 计算 CHUNK_M / CHUNK_N（UB 切分）
    //    kernel 侧按 kBlk=min(32, K) 分块加载，UB buffer 尺寸与 K 无关，
    //    故 chunk 上限由 AllocBuffers 的 UB 预算决定，不受 K 压缩。
    //    AllocBuffers 的 13 个 TBuf 精确核算（chunkM=16, chunkN=192, kBlk=32）：
    //      calc   = mAl*nAl*4 = 12288B  × 5 (acc/t1/t2/t3/y) = 61440
    //      x1Buf/x1fBuf = mAl*kAl*4 = 2048B  × 2            =  4096
    //      x2Buf/x2fBuf = kAl*nAl*4 = 24576B  × 2            = 49152
    //      x2hBuf = 2*max(kAl*nAl, mAl*nAl) = 24576B         = 24576
    //      x1hBuf = 2*max(mAl*kAl, mAl*nAl) = 12288B         = 12288
    //      redBuf/statBuf = 512B           × 2               =  1024
    //      合计 152576B ≈ 149KB < 192KB（余量 43KB）
    //    chunkN=256 时 calc 与 x2h/x2 同步涨到 32KB，合计 205KB > 192KB，
    //    会触发真机 UB 溢出崩溃（仿真器无硬容量限制，无法暴露），故上限取 192。
    // ------------------------------------------------------------------
    int64_t chunkM = M;
    if (chunkM > 16) { chunkM = 16; }
    if (chunkM < 1) { chunkM = 1; }

    int64_t chunkN = N;
    if (chunkN > 192) { chunkN = 192; }
    if (chunkN < 1) { chunkN = 1; }

    // ------------------------------------------------------------------
    // 3. 多核切分策略
    //    per-tensor 量化需全局 abs_max，须跨核归约。kernel 入口不接收
    //    workspace，无独立 GM 归约区；故复用 y GM 前 blockDim 个 float
    //    位作每核的 local abs_max 临时区，双 SyncAll 完成
    //    "写 local → 读全部 → 求 max" 的握手。
    //    按 M 块（chunkM）轮询分配：每核负责全部 N 列、行按 chunkM 步长
    //    跨核取，负载最均衡且 x1Scale/x2Scale 只需读一次。
    //    归约区为 y GM 前 blockDim 个 float（= blockDim*8 个 int8 位），
    //    故 totalNum 须 ≥ blockDim*8。
    // ------------------------------------------------------------------
    int64_t totalMChunks = CeilDiv(M, chunkM);
    int64_t blockDim = totalMChunks;
    if (blockDim > coreNum) { blockDim = coreNum; }
    if (blockDim < 1) { blockDim = 1; }
    // 归约区借用 y GM 前 blockDim 个 float：要求 y 至少有 blockDim*8 个 int8 元素
    while (blockDim > 1 && M * N < blockDim * 8) { blockDim--; }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // 4. 打包 TilingData
    // ------------------------------------------------------------------
    int64_t totalNum = M * N;
    if (totalNum <= 0) { totalNum = 1; }

    uint64_t blockPacked = (static_cast<uint64_t>(M) & M_MASK) |
        ((static_cast<uint64_t>(K) & K_MASK) << 24) |
        ((transposeX1 ? 1ULL : 0ULL) << 48) |
        ((transposeX2 ? 1ULL : 0ULL) << 49);

    uint64_t ubPacked = (static_cast<uint64_t>(N) & N_MASK) |
        ((static_cast<uint64_t>(chunkM) & CHUNK_M_MASK) << 24) |
        ((static_cast<uint64_t>(chunkN) & CHUNK_N_MASK) << 36);

    tiling->totalNum = totalNum;
    tiling->blockFactor = static_cast<int64_t>(blockPacked);
    tiling->ubFactor = static_cast<int64_t>(ubPacked);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(QUANTMATMULSIGMOIDMULQUANT_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(QUANTMATMULSIGMOIDMULQUANT_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantMatmulSigmoidMulQuant([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantMatmulSigmoidMulQuantCompileInfo {};

IMPL_OP_OPTILING(QuantMatmulSigmoidMulQuant).Tiling(QuantMatmulSigmoidMulQuantTilingFunc).TilingParse<QuantMatmulSigmoidMulQuantCompileInfo>(TilingParseForQuantMatmulSigmoidMulQuant);

} // namespace optiling
