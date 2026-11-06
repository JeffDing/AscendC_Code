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
    // 输入：
    //   x1: (M, K) int8
    //   x2: (K, N) int8
    //   scale: (N) float
    //   pertokenScale: (M) float
    // 输出：
    //   out: (M, N) bf16
    // 本算子无 transpose attrs。
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);

    int64_t M = 1, N = 1;
    if (x1Shape != nullptr) {
        if (x1Shape->GetDimNum() >= 2) {
            M = x1Shape->GetDim(0);
        }
    }
    if (x2Shape != nullptr) {
        if (x2Shape->GetDimNum() >= 2) {
            N = x2Shape->GetDim(1);
        } else if (x2Shape->GetDimNum() == 1) {
            N = x2Shape->GetDim(0);
        }
    }
    if (M <= 0) { M = 1; }
    if (N <= 0) { N = 1; }

    // output 0: out (M, N)
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(2);
    outShape->SetDim(0, M);
    outShape->SetDim(1, N);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBatchMatMulV3).InferShape(InferShapeQuantBatchMatMulV3);

} // namespace ops
