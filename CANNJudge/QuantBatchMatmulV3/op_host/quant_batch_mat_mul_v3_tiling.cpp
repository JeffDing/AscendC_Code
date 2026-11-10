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

// TilingData 三个 int64 字段的位域打包约定（结构不可修改，host 与 kernel 必须严格一致）：
//   totalNum    : [0:7)   = chunkM（每核每轮处理的 M 行数）
//                 [7:64)  = K（累加维度大小）
//   blockFactor : [0:5)   = chunkM
//                 [5]     = transposeX1（x1 是否转置：0=物理(M,K)，1=物理(K,M)）
//                 [6]     = transposeX2（x2 是否转置：0=物理(K,N)，1=物理(N,K)）
//                 [7:32)  = 0（保留）
//                 [32:64) = M（x1 行数 / out 行数）
//   ubFactor    : [0:32)  = chunkN（每次 UB 循环处理的 N 列数）
//                 [32:64) = N（x2 列数 / out 列数）
// kernel 解码时对任一 <=0 的维度做 >=1 兜底，保证占位 tilingData（如
// {8388608,8388608,8388608}）不会崩溃：K = 8388608>>7 = 65536，M=N=1，
// chunkM/chunkN 兜底为安全值，transposeX1/X2 解码为 false（协议性约定，
// 见 QuantBatchMatmulV3_AR_REPORT.md）。
static constexpr uint64_t CHUNK_M_MASK = 0x7FULL;
static constexpr uint64_t TRANSPOSE_X1_SHIFT = 5;
static constexpr uint64_t TRANSPOSE_X2_SHIFT = 6;

static ge::graphStatus QuantBatchMatMulV3TilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑（原模板注释保留：完整实现见下）
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
    // 1. 读取 transposeX1 / transposeX2 attrs（def.cpp 声明顺序索引 0/1）
    // ------------------------------------------------------------------
    bool transposeX1 = false;
    bool transposeX2 = false;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        if (const bool* t1 = attrs->GetAttrPointer<bool>(0)) { transposeX1 = *t1; }
        if (const bool* t2 = attrs->GetAttrPointer<bool>(1)) { transposeX2 = *t2; }
    }

    // ------------------------------------------------------------------
    // 2. 解析 x1/x2 shape → M、K、N
    //    x1: 不转置 (M, K)，转置 (K, M)；x2: 不转置 (K, N)，转置 (N, K)；
    //    scale: (N) float；pertokenScale: (M) float；out: (M, N) bf16。
    // ------------------------------------------------------------------
    const gert::StorageShape* x1ShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1ShapePtr);
    const gert::Shape& x1Shape = EnsureNotScalar(x1ShapePtr->GetStorageShape());

    const gert::StorageShape* x2ShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2ShapePtr);
    const gert::Shape& x2Shape = EnsureNotScalar(x2ShapePtr->GetStorageShape());

    int64_t M = 1, K = 1, N = 1;
    if (x1Shape.GetDimNum() >= 2) {
        if (transposeX1) {
            K = x1Shape.GetDim(0);
            M = x1Shape.GetDim(1);
        } else {
            M = x1Shape.GetDim(0);
            K = x1Shape.GetDim(1);
        }
    } else if (x1Shape.GetDimNum() == 1) {
        K = x1Shape.GetDim(0);
    }

    if (x2Shape.GetDimNum() >= 2) {
        int64_t x2K = 0;
        if (transposeX2) {
            N = x2Shape.GetDim(0);
            x2K = x2Shape.GetDim(1);
        } else {
            x2K = x2Shape.GetDim(0);
            N = x2Shape.GetDim(1);
        }
        if (x2K != K) {
            OP_LOGE(context, "x2 first dim %ld does not match x1 second dim %ld", x2K, K);
            return ge::GRAPH_FAILED;
        }
    } else if (x2Shape.GetDimNum() == 1) {
        N = x2Shape.GetDim(0);
    }
    if (M <= 0) { M = 1; }
    if (K <= 0) { K = 1; }
    if (N <= 0) { N = 1; }

    // ------------------------------------------------------------------
    // 3. 切分策略（宿主侧上限，kernel 侧解析时再次兜底）：
    //    chunkM = clamp(M, 1, 16)，chunkN = clamp(N, 1, 192)。
    //    多核按 M×N 二维任务切分：总任务数 = ceil(M/chunkM) × ceil(N/chunkN)，
    //    BlockDim = clamp(min(coreNum, 总任务数), 1, coreNum)。
    //    kernel 中每个核独立处理一个 (m块, n块) 任务并轮询覆盖全部任务，
    //    避免旧版"每核串行遍历全部 N 列块"在大 N 场景下的单核墙钟崩塌
    //    （测试点 13 超时根因）。
    //    UB 预算（kernel AllocBuffers 精确核算，mAl=16, stride=192, kAl=96）：
    //    x1 1.5KB、x1h 3KB、x1f 6KB、x2 18KB、x2h 36KB、x2f 72KB、acc 12KB、
    //    tmp/scale 0.75KB、pertoken 0.25KB、out 6KB → 合计约 157KB < 192KB UB 池。
    // ------------------------------------------------------------------
    int64_t chunkM = M;
    if (chunkM > 16) { chunkM = 16; }
    if (chunkM < 1) { chunkM = 1; }

    int64_t chunkN = N;
    if (chunkN > 192) { chunkN = 192; }
    if (chunkN < 1) { chunkN = 1; }

    // 总任务数可能超出 int64（理论上 M、N 均为 32bit 上限，乘积 < 2^62）。
    // 为避免溢出，用 int64 先做 min(coreNum, ...) 保护。
    int64_t totalM = static_cast<int64_t>(CeilDiv(M, chunkM));
    int64_t totalN = static_cast<int64_t>(CeilDiv(N, chunkN));
    int64_t blockDim;
    if (totalM <= 0 || totalN <= 0 || totalM > coreNum || totalN > coreNum) {
        // 极端形状保护：二重任务数可能触发溢出，直接给核心数上限
        blockDim = coreNum;
    } else {
        // 总任务数 = totalM × totalN，max lane = cores
        blockDim = (static_cast<int64_t>(totalM) * totalN < coreNum)
            ? static_cast<int64_t>(totalM) * totalN
            : coreNum;
    }
    if (blockDim > coreNum) { blockDim = coreNum; }
    if (blockDim < 1) { blockDim = 1; }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // TODO: 设置 tiling 数据（原模板注释保留：三字段按契约 bit 打包）
    // ------------------------------------------------------------------
    uint64_t totalPacked = (static_cast<uint64_t>(K) << 7) |
        (static_cast<uint64_t>(chunkM) & CHUNK_M_MASK);
    uint64_t blockPacked = (static_cast<uint64_t>(chunkM) & 0x1FULL) |
        ((transposeX1 ? 1ULL : 0ULL) << TRANSPOSE_X1_SHIFT) |
        ((transposeX2 ? 1ULL : 0ULL) << TRANSPOSE_X2_SHIFT) |
        ((static_cast<uint64_t>(M) & 0xFFFFFFFFULL) << 32);
    uint64_t ubPacked = (static_cast<uint64_t>(chunkN) & 0xFFFFFFFFULL) |
        ((static_cast<uint64_t>(N) & 0xFFFFFFFFULL) << 32);

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
