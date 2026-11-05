/*!
 * \file depthwise_conv2_d.cpp
 * \brief DepthwiseConv2D 算子 kernel 入口
 */

#include "depthwise_conv2_d.h"

enum class DepthwiseConv2DTilingKey : uint32_t
{
    TILING_KEY_DEPTHWISECONV2D_MODE_0 = 0,
    TILING_KEY_DEPTHWISECONV2D_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void depthwise_conv2_d(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DepthwiseConv2DTilingData);
    GET_TILING_DATA_WITH_STRUCT(DepthwiseConv2DTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(DepthwiseConv2DTilingKey::TILING_KEY_DEPTHWISECONV2D_MODE_0)) {
        NsDepthwiseConv2D::DepthwiseConv2D<half> op;
        op.Init(x, w, b, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(DepthwiseConv2DTilingKey::TILING_KEY_DEPTHWISECONV2D_MODE_1)) {
        NsDepthwiseConv2D::DepthwiseConv2D<float> op;
        op.Init(x, w, b, y, &tilingData);
        op.Process();
    }
}
