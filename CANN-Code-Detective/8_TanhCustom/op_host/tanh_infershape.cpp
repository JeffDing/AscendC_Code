/*!
 * \file tanh_infershape.cpp
 * \brief Tanh 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTanh(gert::InferShapeContext* context)
{
    // Tanh 为 element-wise 算子：输出 shape 与输入 shape 一致
    const gert::Shape* input_shape = (1 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape

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

IMPL_OP_INFERSHAPE(Tanh).InferShape(InferShapeTanh);

} // namespace ops
