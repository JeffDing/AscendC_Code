/*!
 * \file quant_matmul_sigmoid_mul_quant_infershape.cpp
 * \brief QuantMatmulSigmoidMulQuant 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeQuantMatmulSigmoidMulQuant(gert::InferShapeContext* context)
{
    // 输入：
    //   x1: INT8, transposeX1=False 时 (M, K)，True 时 (K, M)
    //   x2: INT8, transposeX2=False 时 (K, N)，True 时 (N, K)
    //   x1Scale/x2Scale: FLOAT (1,)
    //   bias: FLOAT (optional)
    // 输出：
    //   y: INT8 (M, N)
    //   yScale: FLOAT (1,)
    // 属性：
    //   transposeX1 (index 0), transposeX2 (index 1)

    bool transposeX1 = false;
    bool transposeX2 = false;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* t1Ptr = attrs->GetAttrPointer<bool>(0);
        if (t1Ptr != nullptr) {
            transposeX1 = *t1Ptr;
        }
        const bool* t2Ptr = attrs->GetAttrPointer<bool>(1);
        if (t2Ptr != nullptr) {
            transposeX2 = *t2Ptr;
        }
    }

    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);

    int64_t M = 1;
    int64_t K = 1;
    int64_t N = 1;

    if (x1Shape != nullptr && x1Shape->GetDimNum() >= 2) {
        if (transposeX1) {
            K = x1Shape->GetDim(0);
            M = x1Shape->GetDim(1);
        } else {
            M = x1Shape->GetDim(0);
            K = x1Shape->GetDim(1);
        }
    }
    if (x2Shape != nullptr && x2Shape->GetDimNum() >= 2) {
        if (transposeX2) {
            N = x2Shape->GetDim(0);
        } else {
            N = x2Shape->GetDim(1);
        }
    }
    if (M <= 0) { M = 1; }
    if (K <= 0) { K = 1; }
    if (N <= 0) { N = 1; }

    // output 0: y (M, N)
    gert::Shape* yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    yShape->SetDimNum(2);
    yShape->SetDim(0, M);
    yShape->SetDim(1, N);

    // output 1: yScale (1,)
    gert::Shape* yScaleShape = context->GetOutputShape(1);
    if (yScaleShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    yScaleShape->SetDimNum(1);
    yScaleShape->SetDim(0, 1);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantMatmulSigmoidMulQuant).InferShape(InferShapeQuantMatmulSigmoidMulQuant);

} // namespace ops
