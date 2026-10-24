/*!
 * \file mse_loss_infershape.cpp
 * \brief MseLoss 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMseLoss(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    const gert::Shape* input_shape = (2 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape

    // 读取 reduction 属性: 0=none(输出与 self 同 shape), 1=mean / 2=sum(输出标量 shape [1])
    int64_t reduction = 1;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* reductionPtr = attrs->GetAttrPointer<int64_t>(0);
        if (reductionPtr != nullptr) {
            reduction = *reductionPtr;
        }
    }

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (reduction == 0 && input_shape != nullptr) {
            // none: 输出与 self 同 shape
            *output_shape = *input_shape;
        } else {
            // mean / sum: 输出标量 shape [1]
            output_shape->SetDimNum(1);
            output_shape->SetDim(0, 1);
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MseLoss).InferShape(InferShapeMseLoss);

} // namespace ops
