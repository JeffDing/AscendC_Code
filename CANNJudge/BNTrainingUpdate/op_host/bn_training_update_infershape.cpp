/*!
 * \file bn_training_update_infershape.cpp
 * \brief BnTrainingUpdate 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeBnTrainingUpdate(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    // 输入：x(N, C, H, W)、sum(C,)、square_sum(C,)、scale(C,)、offset(C,)
    // 输出：y(N, C, H, W)、save_mean(C,)、save_variance(C,)

    const gert::Shape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // y 与 x 同 shape
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape;

    // save_mean / save_variance 均为 (C,)，优先直接采用统计量输入的 shape
    gert::Shape* mean_shape = context->GetOutputShape(1);
    gert::Shape* var_shape = context->GetOutputShape(2);
    if (mean_shape == nullptr || var_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* stat_shape = context->GetInputShape(1);
    if (stat_shape != nullptr && stat_shape->GetDimNum() == 1) {
        *mean_shape = *stat_shape;
        *var_shape = *stat_shape;
    } else {
        // 退化场景：由 x 的第 1 维（通道维）推导
        int64_t channel = (x_shape->GetDimNum() >= 2) ? x_shape->GetDim(1) : 1;
        if (channel <= 0) {
            channel = 1;
        }
        mean_shape->SetDimNum(1);
        mean_shape->SetDim(0, channel);
        var_shape->SetDimNum(1);
        var_shape->SetDim(0, channel);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BnTrainingUpdate).InferShape(InferShapeBnTrainingUpdate);

} // namespace ops