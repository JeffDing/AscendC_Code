/*!
 * \file flash_attention_tiling.cpp
 * \brief FlashAttention 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/flash_attention_tiling_data.h"
#include "../op_kernel/flash_attention_tiling_key.h"

#include <cmath>
#include <cstring>
#include <cstdint>

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

// ---------------------------------------------------------------------------
// TilingData 仅有 3 个 int64 字段（结构不可修改），故将 FlashAttention 运行期
// 所需的全部元信息按位打包进这 3 个字段：
//   totalNum    = D | (S << 10) | (N << 23) | (B << 29) | (causal << 35) | (mask << 36)
//   blockFactor = softmax_scale 的 fp32 位模式（低 32bit，作为 uint32 存取）
//   ubFactor    = 保留（0）
// kernel 端 flash_attention.h 中按同样的布局解码，两者必须保持一致。
// ---------------------------------------------------------------------------
static constexpr int64_t FA_D_BITS = 10;
static constexpr int64_t FA_S_BITS = 13;
static constexpr int64_t FA_N_BITS = 6;
static constexpr int64_t FA_B_BITS = 6;
static constexpr int64_t FA_CAUSAL_BIT = 35;
static constexpr int64_t FA_MASK_BIT = 36;

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

static ge::graphStatus FlashAttentionTilingFunc(gert::TilingContext* context)
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

    FlashAttentionTilingData* tiling = context->GetTilingData<FlashAttentionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 解析 q 的形状 (B, N, S, D)
    int64_t b = 1;
    int64_t n = 1;
    int64_t s = 1;
    int64_t d = 1;
    const auto qShapePtr = context->GetInputShape(0);
    if (qShapePtr != nullptr) {
        const gert::Shape& sh = qShapePtr->GetStorageShape();
        const size_t dn = sh.GetDimNum();
        if (dn >= 4) {
            b = sh.GetDim(dn - 4);
            n = sh.GetDim(dn - 3);
            s = sh.GetDim(dn - 2);
            d = sh.GetDim(dn - 1);
        } else if (dn == 3) {
            n = sh.GetDim(0);
            s = sh.GetDim(1);
            d = sh.GetDim(2);
        } else if (dn == 2) {
            s = sh.GetDim(0);
            d = sh.GetDim(1);
        } else if (dn == 1) {
            s = sh.GetDim(0);
        }
    }
    if (b < 1) { b = 1; }
    if (n < 1) { n = 1; }
    if (s < 1) { s = 1; }
    if (d < 1) { d = 1; }

    // 属性：softmax_scale / is_causal
    float scale = 0.0f;
    bool isCausal = false;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* scalePtr = attrs->GetAttrPointer<float>(0);
        if (scalePtr != nullptr) {
            scale = *scalePtr;
        }
        const bool* causalPtr = attrs->GetAttrPointer<bool>(1);
        if (causalPtr != nullptr) {
            isCausal = *causalPtr;
        }
    }
    if (scale <= 0.0f) {
        scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));
    }

    // 可选输入 attn_mask (B, 1, S, S)：dimNum >= 3 视为已提供
    bool hasMask = false;
    const auto maskShapePtr = context->GetInputShape(3);
    if (maskShapePtr != nullptr && maskShapePtr->GetStorageShape().GetDimNum() >= 3) {
        hasMask = true;
    }

    // dtype: fp16/bf16 走 half kernel，其余走 float kernel
    bool isHalf = false;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
                                 inputDesc->GetDataType() == ge::DT_BF16)) {
        isHalf = true;
    }

    // TODO: 设置 tiling 数据
    // 打包维度与属性到 totalNum / blockFactor
    int64_t packed = 0;
    packed |= (d & 0x3FFLL);
    packed |= ((s & 0x1FFFLL) << FA_D_BITS);
    packed |= ((n & 0x3FLL) << (FA_D_BITS + FA_S_BITS));
    packed |= ((b & 0x3FLL) << (FA_D_BITS + FA_S_BITS + FA_N_BITS));
    if (isCausal) {
        packed |= (1LL << FA_CAUSAL_BIT);
    }
    if (hasMask) {
        packed |= (1LL << FA_MASK_BIT);
    }
    uint32_t scaleBits = 0U;
    std::memcpy(&scaleBits, &scale, sizeof(float));

    tiling->totalNum = packed;
    tiling->blockFactor = static_cast<int64_t>(static_cast<uint64_t>(scaleBits));
    tiling->ubFactor = 0;

    // 多核切分：以 (b, n, 行) 为任务粒度，每个任务独立处理一行 query
    int64_t tasks = b * n * s;
    int64_t usedCore = (coreNum < tasks) ? coreNum : tasks;
    if (usedCore < 1) {
        usedCore = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalf) {
        tilingKey = GET_TPL_TILING_KEY(FLASHATTENTION_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(FLASHATTENTION_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFlashAttention([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FlashAttentionCompileInfo {};

IMPL_OP_OPTILING(FlashAttention).Tiling(FlashAttentionTilingFunc).TilingParse<FlashAttentionCompileInfo>(TilingParseForFlashAttention);

} // namespace optiling