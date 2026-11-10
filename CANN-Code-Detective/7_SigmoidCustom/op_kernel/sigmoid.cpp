/*!
 * \file sigmoid.cpp
 * \brief Sigmoid 算子 kernel 入口
 */

#include "sigmoid.h"

enum class SigmoidTilingKey : uint32_t
{
    TILING_KEY_SIGMOID_MODE_0 = 0,
    TILING_KEY_SIGMOID_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidTilingData);
    GET_TILING_DATA_WITH_STRUCT(SigmoidTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SigmoidTilingKey::TILING_KEY_SIGMOID_MODE_0)) {
        NsSigmoid::Sigmoid<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SigmoidTilingKey::TILING_KEY_SIGMOID_MODE_1)) {
        NsSigmoid::Sigmoid<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
