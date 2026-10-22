/*!
 * \file confusion_mul_grad_infershape.cpp
 * \brief ConfusionMulGrad 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeConfusionMulGrad(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    const gert::Shape* input_shape = (3 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // output0 = input0 ⊙ input1, 与输入同 shape
    gert::Shape* output0_shape = context->GetOutputShape(0);
    if (output0_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *output0_shape = *input_shape;

    // 解析 (M, N), 退化场景按 1 处理
    int64_t m = 1;
    int64_t n = 1;
    if (input_shape->GetDimNum() >= 2) {
        m = input_shape->GetDim(0);
        n = input_shape->GetDim(1);
    } else if (input_shape->GetDimNum() == 1) {
        m = input_shape->GetDim(0);
    }
    if (m < 1) {
        m = 1;
    }
    if (n < 1) {
        n = 1;
    }

    // axis 属性(list_int)归一化为 2D 语义, keep_dims 属性(bool, 缺省 false)
    bool hasAxis[2] = {false, false};
    bool keepDims = false;
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
        const bool* keepPtr = attrs->GetBool(1);
        if (keepPtr != nullptr) {
            keepDims = *keepPtr;
        }
    }

    // output1 = ReduceSum(input1 ⊙ input2, axis, keep_dims)
    gert::Shape* output1_shape = context->GetOutputShape(1);
    if (output1_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (hasAxis[0] && hasAxis[1]) {
        // 全归约 axis=[0,1]: keep_dims -> (1,1), 否则标量
        if (keepDims) {
            output1_shape->SetDimNum(2);
            output1_shape->SetDim(0, 1);
            output1_shape->SetDim(1, 1);
        } else {
            output1_shape->SetDimNum(0);
        }
    } else if (hasAxis[0]) {
        // 列归约 axis=[0]: 输出长度 N
        if (keepDims) {
            output1_shape->SetDimNum(2);
            output1_shape->SetDim(0, 1);
            output1_shape->SetDim(1, n);
        } else {
            output1_shape->SetDimNum(1);
            output1_shape->SetDim(0, n);
        }
    } else {
        // 行归约 axis=[1](空 axis 防御时与 tiling 端约定一致按全归约处理, 此处
        // hasAxis[1] 为 true 的正常场景): 输出长度 M
        if (hasAxis[1]) {
            if (keepDims) {
                output1_shape->SetDimNum(2);
                output1_shape->SetDim(0, m);
                output1_shape->SetDim(1, 1);
            } else {
                output1_shape->SetDimNum(1);
                output1_shape->SetDim(0, m);
            }
        } else {
            if (keepDims) {
                output1_shape->SetDimNum(2);
                output1_shape->SetDim(0, 1);
                output1_shape->SetDim(1, 1);
            } else {
                output1_shape->SetDimNum(0);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConfusionMulGrad).InferShape(InferShapeConfusionMulGrad);

} // namespace ops
