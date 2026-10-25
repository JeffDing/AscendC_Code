/*!
 * \file mhc_sinkhorn_grad_infershape.cpp
 * \brief MhcSinkhornGrad 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMhcSinkhornGrad(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    // gradInput 与 gradOutput 同形：(T, n, n) 或 (B, S, n, n)
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output_shape = *input_shape;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcSinkhornGrad).InferShape(InferShapeMhcSinkhornGrad);

} // namespace ops
