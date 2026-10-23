/*!
 * \file group_norm_silu_quant_tiling.cpp
 * \brief GroupNormSiluQuant 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/group_norm_silu_quant_tiling_data.h"
#include "../op_kernel/group_norm_silu_quant_tiling_key.h"

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
//   blockFactor : [0:32)  = hwSize（x 中 C 之后所有维度的乘积）
//                 [32:48) = C（通道数）
//                 [48:64) = G（num_groups，分组数）
//   ubFactor    : [0:32)  = epsilon 的 IEEE-754 位模式
//                 [32:62) = N（batch 数）
//                 [62]    = activateSilu（1 = 开启 SiLU）
//                 [63]    = quantScale 是否按通道（1 = shape (C,)，0 = 标量）
//   totalNum    : 保留 = N * C * hwSize（kernel 侧不读取该字段，仅作元素总数记录）
constexpr uint64_t HWSIZE_MASK = 0xFFFFFFFFULL;  // 32 bit
constexpr uint64_t C_MASK = 0xFFFFULL;           // 16 bit
constexpr uint64_t G_MASK = 0xFFFFULL;           // 16 bit
constexpr uint64_t N_MASK = 0x3FFFFFFFULL;       // 30 bit

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

static ge::graphStatus GroupNormSiluQuantTilingFunc(gert::TilingContext* context)
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

    GroupNormSiluQuantTilingData* tiling = context->GetTilingData<GroupNormSiluQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // TODO: 设置 tiling 数据
    // ------------------------------------------------------------------
    // 1. 解析 self 的 (N, C, *)，兼容更低维度的退化输入
    // ------------------------------------------------------------------
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    int64_t n = 1;
    int64_t c = 1;
    int64_t hwSize = 1;
    size_t dimNum = xShape.GetDimNum();
    if (dimNum >= 2) {
        n = xShape.GetDim(0);
        c = xShape.GetDim(1);
        for (size_t i = 2; i < dimNum; ++i) {
            int64_t d = xShape.GetDim(i);
            if (d <= 0) { d = 1; }
            hwSize *= d;
        }
    } else if (dimNum == 1) {
        c = xShape.GetDim(0);
    }
    if (n <= 0) { n = 1; }
    if (hwSize <= 0) { hwSize = 1; }

    // 用 gamma 的 shape 校正通道数（gamma / beta 均为 (C,)）
    const gert::StorageShape* gammaShapePtr = context->GetInputShape(1);
    if (gammaShapePtr != nullptr) {
        const gert::Shape& gammaShape = gammaShapePtr->GetStorageShape();
        if (gammaShape.GetDimNum() >= 1 && gammaShape.GetDim(0) > 0) {
            c = gammaShape.GetDim(0);
        }
    }
    if (c <= 0) { c = 1; }

    int64_t totalNum = n * c * hwSize;    // 元素总数
    if (totalNum <= 0) { totalNum = 1; }

    // quantScale（input index 3）：元素数 > 1 视为按通道 (C,)，否则为标量
    int64_t qsNum = 1;
    const gert::StorageShape* qsShapePtr = context->GetInputShape(3);
    if (qsShapePtr != nullptr) {
        const gert::Shape& qsShape = EnsureNotScalar(qsShapePtr->GetStorageShape());
        int64_t prod = 1;
        for (size_t i = 0; i < qsShape.GetDimNum(); ++i) {
            int64_t d = qsShape.GetDim(i);
            prod *= (d > 0) ? d : 1;
        }
        if (prod > 0) { qsNum = prod; }
    }
    bool qsPerChannel = (qsNum > 1);

    // ------------------------------------------------------------------
    // 2. 解析属性：group（必选，index 0）、eps（可选，index 1，默认 1e-5）、
    //    activateSilu（可选，index 2，默认 true）
    // ------------------------------------------------------------------
    int64_t numGroups = 1;
    float epsilon = 1e-5f;
    bool activateSilu = true;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* gPtr = attrs->GetAttrPointer<int64_t>(0);
        if (gPtr != nullptr) {
            numGroups = *gPtr;
        }
        const float* epsPtr = attrs->GetAttrPointer<float>(1);
        if (epsPtr != nullptr) {
            epsilon = *epsPtr;
        }
        const bool* siluPtr = attrs->GetAttrPointer<bool>(2);
        if (siluPtr != nullptr) {
            activateSilu = *siluPtr;
        }
    }
    if (numGroups <= 0) { numGroups = 1; }
    if (numGroups > c) { numGroups = c; }
    if (!(epsilon > 0.0f)) { epsilon = 1e-5f; }

    // ------------------------------------------------------------------
    // 3. 多核切分：按组（N * G 个组）均分，组间归约互相独立，无跨核依赖
    // ------------------------------------------------------------------
    int64_t totalGroups = n * numGroups;
    int64_t blockDim = coreNum;
    if (totalGroups < blockDim) {
        blockDim = totalGroups;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // ------------------------------------------------------------------
    // 4. UB 切分：kernel 侧使用编译期固定的单块元素数，恒不超过 UB 容量，
    //    无需 host 下发
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // 5. 打包 TilingData（结构不可改，字段语义与 kernel 注释保持一致）
    // ------------------------------------------------------------------
    union { uint32_t u; float f; } epsConv;
    epsConv.f = epsilon;
    uint32_t epsBits = epsConv.u;

    uint64_t ubPacked = static_cast<uint64_t>(epsBits) |
        ((static_cast<uint64_t>(n) & N_MASK) << 32) |
        ((activateSilu ? 1ULL : 0ULL) << 62) |
        ((qsPerChannel ? 1ULL : 0ULL) << 63);

    tiling->totalNum = totalNum;
    tiling->blockFactor = static_cast<int64_t>(
        (static_cast<uint64_t>(hwSize) & HWSIZE_MASK) |
        ((static_cast<uint64_t>(c) & C_MASK) << 32) |
        ((static_cast<uint64_t>(numGroups) & G_MASK) << 48));
    tiling->ubFactor = static_cast<int64_t>(ubPacked);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(GROUPNORMSILUQUANT_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(GROUPNORMSILUQUANT_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGroupNormSiluQuant([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GroupNormSiluQuantCompileInfo {};

IMPL_OP_OPTILING(GroupNormSiluQuant).Tiling(GroupNormSiluQuantTilingFunc).TilingParse<GroupNormSiluQuantCompileInfo>(TilingParseForGroupNormSiluQuant);

} // namespace optiling
