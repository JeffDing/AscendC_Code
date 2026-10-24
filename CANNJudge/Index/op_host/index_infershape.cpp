/*!
 * \file index_infershape.cpp
 * \brief Index 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/tensor.h"

#include <vector>

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeIndex(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    const gert::Shape* input_shape = (3 > 0) ? context->GetInputShape(0) : nullptr;
    // 注意：无输入算子时 input_shape 为 nullptr，需在此处手动设置输出 shape

    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* maskShape = context->GetInputShape(1);
    const gert::Shape* indicesShape = context->GetInputShape(2);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || indicesShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t ndim = static_cast<int64_t>(xShape->GetDimNum());
    if (ndim <= 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t maskNum = ndim;
    if (maskShape != nullptr && maskShape->GetDimNum() == 1) {
        maskNum = maskShape->GetDim(0);
    }

    // 读取 indexed_sizes 的常量值（value depend），确定哪些维度参与索引
    std::vector<int64_t> mask(static_cast<size_t>(ndim), 0);
    bool gotMask = false;
    const gert::Tensor* maskTensor = context->GetInputTensor(1);
    if (maskTensor != nullptr) {
        const int64_t* maskData = maskTensor->GetData<int64_t>();
        if (maskData != nullptr && maskTensor->GetShapeSize() >= maskNum) {
            int64_t cnt = maskNum < ndim ? maskNum : ndim;
            for (int64_t i = 0; i < cnt; ++i) {
                mask[static_cast<size_t>(i)] = maskData[i];
            }
            gotMask = true;
        }
    }

    std::vector<int64_t> idxDims;
    for (size_t i = 0; i < indicesShape->GetDimNum(); ++i) {
        idxDims.push_back(indicesShape->GetDim(i));
    }

    yShape->SetDimNum(0);
    // 无法获取掩码时退化为与输入同形状，保证形状推导不失败
    auto appendXShape = [&]() {
        for (int64_t i = 0; i < ndim; ++i) {
            yShape->AppendDim(xShape->GetDim(i));
        }
    };
    if (!gotMask) {
        appendXShape();
        return ge::GRAPH_SUCCESS;
    }

    std::vector<int64_t> maskedDims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (mask[static_cast<size_t>(i)] != 0) {
            maskedDims.push_back(i);
        }
    }
    if (maskedDims.empty()) {
        appendXShape();
        return ge::GRAPH_SUCCESS;
    }

    int64_t firstM = maskedDims.front();
    int64_t lastM = maskedDims.back();
    bool contiguous = (lastM - firstM + 1 == static_cast<int64_t>(maskedDims.size()));

    if (contiguous) {
        // 连续轴场景：indices 形状插入到被索引的连续维度块位置
        for (int64_t i = 0; i < firstM; ++i) {
            yShape->AppendDim(xShape->GetDim(i));
        }
        for (size_t i = 0; i < idxDims.size(); ++i) {
            yShape->AppendDim(idxDims[i]);
        }
        for (int64_t i = lastM + 1; i < ndim; ++i) {
            yShape->AppendDim(xShape->GetDim(i));
        }
    } else {
        // 非连续轴场景：indices 形状置于最前，其余维度按原顺序保留
        for (size_t i = 0; i < idxDims.size(); ++i) {
            yShape->AppendDim(idxDims[i]);
        }
        for (int64_t i = 0; i < ndim; ++i) {
            bool isMasked = false;
            for (size_t j = 0; j < maskedDims.size(); ++j) {
                if (maskedDims[j] == i) {
                    isMasked = true;
                    break;
                }
            }
            if (!isMasked) {
                yShape->AppendDim(xShape->GetDim(i));
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Index).InferShape(InferShapeIndex);

} // namespace ops