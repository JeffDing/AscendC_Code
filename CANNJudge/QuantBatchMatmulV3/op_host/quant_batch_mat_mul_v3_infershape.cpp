/*!
 * \file quant_batch_mat_mul_v3_infershape.cpp
 * \brief QuantBatchMatMulV3 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeQuantBatchMatMulV3(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    // 输入：
    //   x1: INT8，transposeX1=False 时 (M, K)，True 时 (K, M)
    //   x2: INT8，transposeX2=False 时 (K, N)，True 时 (N, K)
    //   scale: FLOAT (N,)
    //   pertokenScale: FLOAT (M,)
    // 输出：
    //   out: BF16 (M, N)
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

    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    output_shape->SetDimNum(2);
    output_shape->SetDim(0, M);
    output_shape->SetDim(1, N);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBatchMatMulV3).InferShape(InferShapeQuantBatchMatMulV3);

} // namespace ops
