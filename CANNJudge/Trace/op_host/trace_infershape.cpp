/*!
 * \file trace_infershape.cpp
 * \brief Trace 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTrace(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    const gert::Shape* input_shape = (1 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // Trace 输出为标量求和, shape 固定为 (1,)
        (void)input_shape;
        output_shape->SetDimNum(1);
        output_shape->SetDim(0, 1);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Trace).InferShape(InferShapeTrace);

} // namespace ops
