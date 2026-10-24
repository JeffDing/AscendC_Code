/*!
 * \file mse_loss_tiling.cpp
 * \brief MseLoss 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/mse_loss_tiling_data.h"
#include "../op_kernel/mse_loss_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// 每轮 UB 处理元素数: 3 tensor(self/target/out) * 4B * 4096 * double-buffer(2) ≈ 96KB,
// 远低于 192KB 池上限; 且 4096 对齐 32B(float 8 元素 / half 16 元素), 满足 DataCopyPad/ReduceSum 对齐要求
constexpr int64_t MSELOSS_UB_FACTOR = 4096;

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

static ge::graphStatus MseLossTilingFunc(gert::TilingContext* context)
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

    MseLossTilingData* tiling = context->GetTilingData<MseLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 解析输入 shape, 计算总元素数 N(各维度乘积, 退化维度按 1 处理)
    const auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
        int64_t d = inputShape.GetDim(i);
        if (d < 1) {
            d = 1;
        }
        totalNum *= d;
    }
    if (totalNum < 1) {
        totalNum = 1;
    }

    // 读取 reduction 属性: 0=none, 1=mean, 2=sum, 默认 1
    int64_t reduction = 1;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* reductionPtr = attrs->GetAttrPointer<int64_t>(0);
        if (reductionPtr != nullptr) {
            reduction = *reductionPtr;
        }
    }
    if (reduction < 0 || reduction > 2) {
        reduction = 1;
    }

    // dtype: fp16/bf16 走 half kernel, 其余走 float kernel
    bool isHalf = false;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
                                 inputDesc->GetDataType() == ge::DT_BF16)) {
        isHalf = true;
    }

    // 多核切分: 沿元素维均分, 每核至少 1 个元素, 修正核数消除空转核
    int64_t usedCore = (coreNum < totalNum) ? coreNum : totalNum;
    // half + 归约模式(mean/sum): 固定单核, 核内 fp32 精确累加后单次 half 舍入直写
    // (half 多核原子累加存在超精度阈值风险; workspace 汇总在真机 aclnn 路径不可靠, 参考 Trace 经验)
    if (isHalf && reduction != 0) {
        usedCore = 1;
    }
    if (usedCore < 1) {
        usedCore = 1;
    }
    int64_t blockFactor = CeilDiv(totalNum, usedCore);
    usedCore = CeilDiv(totalNum, blockFactor);

    // 每轮 UB 处理元素数, 不超过每核元素数
    // (kernel 侧 buffer 按 CeilAlign(ubFactor, slotElems) 分配, 末轮 cur 非对齐时
    //  用 CeilAlign(cur, slotElems) 处理 DataCopyPad pad 0 的尾部, Sub/Mul/ReduceSum 对齐)
    int64_t ubFactor = MSELOSS_UB_FACTOR;
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    // TODO: 设置 tiling 数据
    // ubFactor 字段高 16 位编码 reduction(none→1, mean→0, sum→2), 低 48 位为每轮 UB 元素数。
    // TilingData 结构不可改(仅 3 字段), 需同时承载 reduction 与 ubFactor;
    // 高位 0 默认 mean, 使 kernel UT 的占位赋值(ubFactor=size, 高位 0)解码为 mean 归约,
    // 与 UT 构造的 outHost(单元素标量)及 golden(mean)天然兼容, 无需修改 test 文件。
    int64_t reductionCode;
    if (reduction == 0) {
        reductionCode = 1;
    } else if (reduction == 2) {
        reductionCode = 2;
    } else {
        reductionCode = 0;
    }
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = (reductionCode << 48) | ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalf) {
        tilingKey = GET_TPL_TILING_KEY(MSELOSS_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(MSELOSS_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMseLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MseLossCompileInfo {};

IMPL_OP_OPTILING(MseLoss).Tiling(MseLossTilingFunc).TilingParse<MseLossCompileInfo>(TilingParseForMseLoss);

} // namespace optiling
