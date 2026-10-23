/*!
 * \file group_norm_silu_quant_infershape.cpp
 * \brief GroupNormSiluQuant 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeGroupNormSiluQuant(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    // 输入：self(N, C, *)、gamma(C,)、beta(C,)、quantScale(1 或 C)
    // 输出：out 与 self 同 shape（int8）；meanOut / rstdOut 为 (N, group)
    const gert::Shape* input_shape = (4 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape

    // group 为必选 Int 属性（attrs index 0），缺失或非法时回退为 1
    int64_t group = 1;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* groupPtr = attrs->GetAttrPointer<int64_t>(0);
        if (groupPtr != nullptr && *groupPtr > 0) {
            group = *groupPtr;
        }
    }

    // batch 数 N：与 tiling 侧解析保持一致，仅当维度 >= 2 时取 dim0，否则回退为 1
    int64_t n = 1;
    if (input_shape != nullptr && input_shape->GetDimNum() >= 2) {
        n = input_shape->GetDim(0);
        if (n <= 0) {
            n = 1;
        }
    }

    for (size_t i = 0; i < 3; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (i == 0) {
            const gert::Shape* in_shape = context->GetInputShape(0);
            if (in_shape == nullptr) {
                in_shape = input_shape;
            }
            if (in_shape != nullptr) {
                *output_shape = *in_shape;
            }
        } else {
            // meanOut / rstdOut：shape 固定为 (N, group)
            output_shape->SetDimNum(2);
            output_shape->SetDim(0, n);
            output_shape->SetDim(1, group);
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSiluQuant).InferShape(InferShapeGroupNormSiluQuant);

} // namespace ops
