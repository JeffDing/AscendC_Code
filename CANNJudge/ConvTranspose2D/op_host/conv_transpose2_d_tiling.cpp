/*!
 * \file conv_transpose2_d_tiling.cpp
 * \brief ConvTranspose2D 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/conv_transpose2_d_tiling_data.h"
#include "../op_kernel/conv_transpose2_d_tiling_key.h"

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

// 读取 int 型属性，兼容「标量 int」与「list int」两种下发形式，缺省时返回默认值。
static int64_t GetIntAttr(const gert::TilingContext* context, size_t index, int64_t defaultValue)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return defaultValue;
    }
    if (const int64_t* v = attrs->GetInt(index)) {
        return *v;
    }
    if (const gert::TypedContinuousVector<int64_t>* v = attrs->GetListInt(index)) {
        if (v->GetSize() > 0) {
            return v->GetData()[0];
        }
    }
    return defaultValue;
}

static int64_t DimAt(const gert::StorageShape* shape, size_t idx, int64_t fallback)
{
    if (shape == nullptr) {
        return fallback;
    }
    const gert::Shape& s = shape->GetStorageShape();
    if (idx >= s.GetDimNum()) {
        return fallback;
    }
    return s.GetDim(idx);
}

// TilingData 位域约定（结构不可改，host / kernel 两侧严格一致）。
// 刻意不把任何关键标量放在 totalNum：真实框架下 TilingData 首字段在 kernel 侧读取不可靠
// （偶发读到 0 / 8，已由常量 tiling 受控实验确证），故 totalNum 仅作冗余信息保留。
//   blockFactor : bit[0:13)  = N
//                 bit[13:26) = C_in
//                 bit[26:39) = C_out
//                 bit[39:50) = H_in
//                 bit[50:61) = W_in
//                 bit[61:64) = 保留
//   ubFactor    : bit[0:4)   = kH
//                 bit[4:8)   = kW
//                 bit[8:20)  = stride
//                 bit[20:32) = padding
//                 bit[32:44) = output_padding
//                 bit[44:56) = dilation
//                 bit[56:57) = hasBias（偏置输入 b 是否有效）
//                 bit[57:64) = 保留
constexpr uint64_t BF_N_SHIFT = 0;
constexpr uint64_t BF_CIN_SHIFT = 13;
constexpr uint64_t BF_COUT_SHIFT = 26;
constexpr uint64_t BF_HIN_SHIFT = 39;
constexpr uint64_t BF_WIN_SHIFT = 50;
constexpr uint64_t BF_N_MASK = 0x1FFFULL;     // 13 bit
constexpr uint64_t BF_CIN_MASK = 0x1FFFULL;   // 13 bit
constexpr uint64_t BF_COUT_MASK = 0x1FFFULL;  // 13 bit
constexpr uint64_t BF_HIN_MASK = 0x7FFULL;    // 11 bit
constexpr uint64_t BF_WIN_MASK = 0x7FFULL;    // 11 bit

constexpr uint64_t UF_KH_SHIFT = 0;
constexpr uint64_t UF_KW_SHIFT = 4;
constexpr uint64_t UF_STRIDE_SHIFT = 8;
constexpr uint64_t UF_PAD_SHIFT = 20;
constexpr uint64_t UF_OPAD_SHIFT = 32;
constexpr uint64_t UF_DIL_SHIFT = 44;
constexpr uint64_t UF_HASBIAS_SHIFT = 56;
constexpr uint64_t UF_K_MASK = 0xFULL;          // 4 bit
constexpr uint64_t UF_PARAM_MASK = 0xFFFULL;    // 12 bit

static ge::graphStatus ConvTranspose2DTilingFunc(gert::TilingContext* context)
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

    ConvTranspose2DTilingData* tiling = context->GetTilingData<ConvTranspose2DTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* wShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);

    // x: (N, C_in, H_in, W_in)，w: (C_in, C_out, kH, kW)
    const int64_t n = DimAt(xShape, 0, 1);
    const int64_t cIn = DimAt(xShape, 1, 1);
    const int64_t hIn = DimAt(xShape, 2, 1);
    const int64_t wIn = DimAt(xShape, 3, 1);
    const int64_t cOut = DimAt(wShape, 1, cIn);
    const int64_t kH = DimAt(wShape, 2, 1);
    const int64_t kW = DimAt(wShape, 3, 1);

    // 属性索引与 op_def 中声明顺序一致
    const int64_t stride = GetIntAttr(context, 0, 1);
    const int64_t padding = GetIntAttr(context, 1, 0);
    const int64_t outputPadding = GetIntAttr(context, 2, 0);
    const int64_t dilation = GetIntAttr(context, 3, 1);

    OP_CHECK_IF(n < 1 || cIn < 1 || cOut < 1 || hIn < 1 || wIn < 1 || kH < 1 || kW < 1,
        OP_LOGE(context, "invalid tensor dim: n=%ld cIn=%ld cOut=%ld hIn=%ld wIn=%ld kH=%ld kW=%ld",
                n, cIn, cOut, hIn, wIn, kH, kW),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(stride < 1 || dilation < 1,
        OP_LOGE(context, "invalid attr: stride=%ld dilation=%ld", stride, dilation),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(padding < 0 || outputPadding < 0,
        OP_LOGE(context, "invalid attr: padding=%ld outputPadding=%ld", padding, outputPadding),
        return ge::GRAPH_FAILED);

    // H_out = (H_in - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1
    const int64_t hOut = (hIn - 1) * stride - 2 * padding + dilation * (kH - 1) + outputPadding + 1;
    const int64_t wOut = (wIn - 1) * stride - 2 * padding + dilation * (kW - 1) + outputPadding + 1;
    OP_CHECK_IF(hOut < 1 || wOut < 1,
        OP_LOGE(context, "invalid output dim: hOut=%ld wOut=%ld", hOut, wOut),
        return ge::GRAPH_FAILED);

    // 位域宽度保护：超限时明确失败，避免静默截断导致 kernel 侧解码错位
    OP_CHECK_IF(n > BF_N_MASK || cIn > BF_CIN_MASK || cOut > BF_COUT_MASK,
        OP_LOGE(context, "dim exceeds bitfield: n=%ld cIn=%ld cOut=%ld", n, cIn, cOut),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(hIn > BF_HIN_MASK || wIn > BF_WIN_MASK,
        OP_LOGE(context, "spatial dim exceeds bitfield: hIn=%ld wIn=%ld", hIn, wIn),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(kH > UF_K_MASK || kW > UF_K_MASK,
        OP_LOGE(context, "kernel dim exceeds bitfield: kH=%ld kW=%ld", kH, kW),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(stride > UF_PARAM_MASK || padding > UF_PARAM_MASK || outputPadding > UF_PARAM_MASK ||
                    dilation > UF_PARAM_MASK,
        OP_LOGE(context, "attr exceeds bitfield: stride=%ld padding=%ld outputPadding=%ld dilation=%ld",
                stride, padding, outputPadding, dilation),
        return ge::GRAPH_FAILED);

    // 偏置输入 b 为 OPTIONAL：未下发时 GetOptionalInputShape 返回 nullptr
    const bool hasBias = (context->GetOptionalInputShape(2) != nullptr);

    // TODO: 设置 tiling 数据
    // 多核切分策略：任务总数 = N * C_out，每个任务负责一个 (n, co) 的完整输出平面
    // (N*C_out, H_out*W_out)。任务间无数据依赖（每个输出元素由唯一的 (n, co, oh, ow) 确定），
    // 因此按任务均分到各核即可，无跨核竞争、无需任何跨核同步。
    const int64_t totalTask = n * cOut;
    int64_t usedCore = coreNum;
    if (usedCore > totalTask) {
        usedCore = totalTask;
    }
    if (usedCore < 1) {
        usedCore = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    uint64_t blockFactor = (static_cast<uint64_t>(n) & BF_N_MASK) << BF_N_SHIFT;
    blockFactor |= (static_cast<uint64_t>(cIn) & BF_CIN_MASK) << BF_CIN_SHIFT;
    blockFactor |= (static_cast<uint64_t>(cOut) & BF_COUT_MASK) << BF_COUT_SHIFT;
    blockFactor |= (static_cast<uint64_t>(hIn) & BF_HIN_MASK) << BF_HIN_SHIFT;
    blockFactor |= (static_cast<uint64_t>(wIn) & BF_WIN_MASK) << BF_WIN_SHIFT;

    uint64_t ubFactor = (static_cast<uint64_t>(kH) & UF_K_MASK) << UF_KH_SHIFT;
    ubFactor |= (static_cast<uint64_t>(kW) & UF_K_MASK) << UF_KW_SHIFT;
    ubFactor |= (static_cast<uint64_t>(stride) & UF_PARAM_MASK) << UF_STRIDE_SHIFT;
    ubFactor |= (static_cast<uint64_t>(padding) & UF_PARAM_MASK) << UF_PAD_SHIFT;
    ubFactor |= (static_cast<uint64_t>(outputPadding) & UF_PARAM_MASK) << UF_OPAD_SHIFT;
    ubFactor |= (static_cast<uint64_t>(dilation) & UF_PARAM_MASK) << UF_DIL_SHIFT;
    ubFactor |= (static_cast<uint64_t>(hasBias ? 1U : 0U)) << UF_HASBIAS_SHIFT;

    // totalNum 承载总任务数作冗余信息（kernel 不依赖，规避首字段不可靠风险）
    tiling->totalNum = totalTask;
    tiling->blockFactor = static_cast<int64_t>(blockFactor);
    tiling->ubFactor = static_cast<int64_t>(ubFactor);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(CONVTRANSPOSE2D_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(CONVTRANSPOSE2D_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForConvTranspose2D([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ConvTranspose2DCompileInfo {};

IMPL_OP_OPTILING(ConvTranspose2D).Tiling(ConvTranspose2DTilingFunc).TilingParse<ConvTranspose2DCompileInfo>(TilingParseForConvTranspose2D);

} // namespace optiling
