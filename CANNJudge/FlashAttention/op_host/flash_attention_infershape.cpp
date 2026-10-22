/*!
 * \file flash_attention_infershape.cpp
 * \brief FlashAttention 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeFlashAttention(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    // FlashAttention 输出 out 的形状与 q 完全一致 (B, N, S, D)，
    // 因此直接将输入 0(q) 的 shape 拷贝到输出 0(out) 即可。
    const gert::Shape* input_shape = (1 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* in_shape = (i < 4) ? context->GetInputShape(i) : input_shape;
        if (in_shape == nullptr) {
            in_shape = input_shape;
        }
        if (in_shape != nullptr) {
            *output_shape = *in_shape;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttention).InferShape(InferShapeFlashAttention);

} // namespace ops