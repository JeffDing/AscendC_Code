/*!
 * \file sigmoid_infershape.cpp
 * \brief Sigmoid 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSigmoid(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* in_shape = (i < 1) ? context->GetInputShape(i) : input_shape;
        if (in_shape == nullptr) {
            in_shape = input_shape;
        }
        if (in_shape != nullptr) {
            *output_shape = *in_shape;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sigmoid).InferShape(InferShapeSigmoid);

} // namespace ops
