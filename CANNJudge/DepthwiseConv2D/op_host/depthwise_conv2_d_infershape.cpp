/*!
 * \file depthwise_conv2_d_infershape.cpp
 * \brief DepthwiseConv2D 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

// 读取 int 型属性，同时兼容「标量 int」与「list int」两种下发形式。
// 属性未设置或越界时返回 defaultValue。
static int64_t GetIntAttr(const gert::InferShapeContext* context, size_t index, int64_t defaultValue)
{
    if (context == nullptr) {
        return defaultValue;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return defaultValue;
    }
    if (const int64_t* v = attrs->GetInt(index)) {
        return *v;
    }
    if (const gert::TypedContinuousVector<int64_t>* v = attrs->GetListInt(index)) {
        if (v->GetSize() > 0) {
            return v->GetData()[0];
        }
    }
    return defaultValue;
}

// 取 shape 第 idx 维；shape 维度不足时返回 fallback（用于兼容非 4 维的退化输入）。
static int64_t DimAt(const gert::Shape* shape, size_t idx, int64_t fallback)
{
    if (shape == nullptr || idx >= shape->GetDimNum()) {
        return fallback;
    }
    return shape->GetDim(idx);
}

static ge::graphStatus InferShapeDepthwiseConv2D(gert::InferShapeContext* context)
{
    // x: (N, C, H, W)，w: (C, 1, kH, kW)，b: (C,) 可选，y: (N, C, H_out, W_out)
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    if (input_shape == nullptr || weight_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t n = DimAt(input_shape, 0, 1);
    const int64_t c = DimAt(input_shape, 1, 1);
    const int64_t hIn = DimAt(input_shape, 2, 1);
    const int64_t wIn = DimAt(input_shape, 3, 1);
    const int64_t kH = DimAt(weight_shape, 2, 1);
    const int64_t kW = DimAt(weight_shape, 3, 1);

    // 属性索引与 op_def 中声明顺序一致：stride / pad / dilation
    const int64_t stride = GetIntAttr(context, 0, 1);
    const int64_t pad = GetIntAttr(context, 1, 0);
    const int64_t dilation = GetIntAttr(context, 2, 1);

    // H_out = (H_in + 2*pad - dilation*(kH-1) - 1) / stride + 1
    const int64_t hOut = (hIn + 2 * pad - dilation * (kH - 1) - 1) / stride + 1;
    const int64_t wOut = (wIn + 2 * pad - dilation * (kW - 1) - 1) / stride + 1;
    if (hOut <= 0 || wOut <= 0 || n <= 0 || c <= 0) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        output_shape->SetDimNum(4);
        output_shape->SetDim(0, n);
        output_shape->SetDim(1, c);
        output_shape->SetDim(2, hOut);
        output_shape->SetDim(3, wOut);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DepthwiseConv2D).InferShape(InferShapeDepthwiseConv2D);

} // namespace ops
