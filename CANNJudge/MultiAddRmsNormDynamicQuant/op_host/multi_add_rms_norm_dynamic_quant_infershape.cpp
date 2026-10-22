/*!
 * \file multi_add_rms_norm_dynamic_quant_infershape.cpp
 * \brief MultiAddRmsNormDynamicQuant 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMultiAddRmsNormDynamicQuant(gert::InferShapeContext* context)
{
    // x1a: (B, H)，其余输入同型或 (H,)
    const gert::Shape* x1aShape = context->GetInputShape(0);
    if (x1aShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape inShape = (x1aShape->GetDimNum() == 0) ? gert::Shape({1}) : *x1aShape;

    // outputs 0~3: y1/y2/x/y 与 x1a 同型 (B, H)
    for (size_t i = 0; i < 4; i++) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = inShape;
    }

    // outputs 4/5: scale1/scale2 为逐行量化 scale，shape 为 (B,)
    gert::Shape bShape;
    if (inShape.GetDimNum() >= 1) {
        bShape.AppendDim(inShape.GetDim(0));
    } else {
        bShape.AppendDim(1);
    }
    for (size_t i = 4; i < 6; i++) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = bShape;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultiAddRmsNormDynamicQuant).InferShape(InferShapeMultiAddRmsNormDynamicQuant);

} // namespace ops
