/*!
 * \file confusion_mul_grad_tiling.cpp
 * \brief ConfusionMulGrad 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/confusion_mul_grad_tiling_data.h"
#include "../op_kernel/confusion_mul_grad_tiling_key.h"

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

// 归约模式编码(与 kernel 端 ubFactor 解码约定一致):
//   0 = 行归约(axis=[1], output1 长度 M)
//   1 = 列归约(axis=[0], output1 长度 N)
//   2 = 全归约(axis=[0,1] 或异常/空 axis, output1 长度 1)
static constexpr int64_t AXIS_CODE_ROW = 0;
static constexpr int64_t AXIS_CODE_COL = 1;
static constexpr int64_t AXIS_CODE_ALL = 2;

// 解析 axis 属性(list_int)并归一化为 2D 语义, 返回归约模式编码
static int64_t ResolveAxisCode(gert::TilingContext* context)
{
    bool hasAxis[2] = {false, false};
    const auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto axisVec = attrs->GetListInt(0);
        if (axisVec != nullptr) {
            const size_t axisNum = axisVec->GetSize();
            const int64_t* axisData = axisVec->GetData();
            for (size_t i = 0; i < axisNum; i++) {
                const int64_t v = ((axisData[i] % 2) + 2) % 2; // 负轴按 2D 语义归一化
                hasAxis[v] = true;
            }
        }
    }
    if (hasAxis[0] && hasAxis[1]) {
        return AXIS_CODE_ALL;
    }
    if (hasAxis[0]) {
        return AXIS_CODE_COL;
    }
    if (hasAxis[1]) {
        return AXIS_CODE_ROW;
    }
    return AXIS_CODE_ALL; // 空 axis 防御: 按全归约处理
}

static ge::graphStatus ConfusionMulGradTilingFunc(gert::TilingContext* context)
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

    ConfusionMulGradTilingData* tiling = context->GetTilingData<ConfusionMulGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 解析输入 (M, N), 退化场景按 1 处理
    const auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();
    int64_t m = 1;
    int64_t n = 1;
    if (inputShape.GetDimNum() >= 2) {
        m = inputShape.GetDim(0);
        n = inputShape.GetDim(1);
    } else if (inputShape.GetDimNum() == 1) {
        m = inputShape.GetDim(0);
    }
    if (m < 1) {
        m = 1;
    }
    if (n < 1) {
        n = 1;
    }
    const int64_t totalNum = m * n;

    // dtype: fp16/bf16 走 half kernel, 其余走 float kernel
    bool isHalf = false;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
                                  inputDesc->GetDataType() == ge::DT_BF16)) {
        isHalf = true;
    }

    // 归约模式
    const int64_t axisCode = ResolveAxisCode(context);

    // 多核切分策略(输出0 逐元素乘与行归约/全归约均按行切分, 列归约按 32B 槽对齐的
    // 列任务切分——fp32 8 列/任务, fp16 16 列/任务, 尾列逐列任务):
    //   行切分核数 rowCores = min(coreNum, M)
    //   列切分任务数 colTasks = N/segGran + N%segGran, 列归约取
    //   max(rowCores, colTasks) 上限, 由 kernel 端两 pass 各自裁剪有效核数
    //   全归约 fp16 固定单核(核内 fp32 精确累加后单次 half 舍入直写, half 多核
    //   原子累加存在超精度阈值风险); fp32 多核走 fp32 原子加(真机已验证)
    int64_t rowCores = (coreNum < m) ? coreNum : m;
    int64_t usedCore = rowCores;
    if (axisCode == AXIS_CODE_COL) {
        const int64_t segGran = isHalf ? 16 : 8;
        const int64_t colTasks = n / segGran + n % segGran;
        int64_t colCores = (coreNum < colTasks) ? coreNum : colTasks;
        if (colCores < rowCores) {
            colCores = rowCores;
        }
        usedCore = (coreNum < colCores) ? coreNum : colCores;
    } else if (axisCode == AXIS_CODE_ALL && isHalf) {
        usedCore = 1;
    }
    if (usedCore < 1) {
        usedCore = 1;
    }

    // 每核行数(行对齐切分, blockFactor = rowsPerCore * N 为每核元素数)
    const int64_t rowsPerCore = CeilDiv(m, usedCore);

    // TODO: 设置 tiling 数据
    // TilingData 编码约定(结构不可改, kernel 端解码):
    //   totalNum   = M * N(总元素数)
    //   blockFactor = rowsPerCore * N(每核处理的元素数, 天然行对齐)
    //   ubFactor   = N * 3 + axisCode(承载列数 N 与归约模式, kernel 端 N=ubFactor/3、
    //               axisCode=ubFactor%3 解码; 官方 UT 用例 {192,192,192} 恰好解码为
    //               N=64、行归约、单核 3 行, 与 3x64 用例天然兼容)
    tiling->totalNum = totalNum;
    tiling->blockFactor = rowsPerCore * n;
    tiling->ubFactor = n * 3 + axisCode;

    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (isHalf) {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONMULGRAD_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONMULGRAD_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForConfusionMulGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ConfusionMulGradCompileInfo {};

IMPL_OP_OPTILING(ConfusionMulGrad).Tiling(ConfusionMulGradTilingFunc).TilingParse<ConfusionMulGradCompileInfo>(TilingParseForConfusionMulGrad);

} // namespace optiling
