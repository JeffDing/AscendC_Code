/*!
 * \file confusion_mul_grad.cpp
 * \brief ConfusionMulGrad 算子 kernel 入口
 */

#include "confusion_mul_grad.h"

enum class ConfusionMulGradTilingKey : uint32_t
{
    TILING_KEY_CONFUSIONMULGRAD_MODE_0 = 0,
    TILING_KEY_CONFUSIONMULGRAD_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void confusion_mul_grad(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR output0, GM_ADDR output1, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ConfusionMulGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ConfusionMulGradTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ConfusionMulGradTilingKey::TILING_KEY_CONFUSIONMULGRAD_MODE_0)) {
        NsConfusionMulGrad::ConfusionMulGrad<half> op;
        op.Init(input0, input1, input2, output0, output1, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ConfusionMulGradTilingKey::TILING_KEY_CONFUSIONMULGRAD_MODE_1)) {
        NsConfusionMulGrad::ConfusionMulGrad<float> op;
        op.Init(input0, input1, input2, output0, output1, &tilingData);
        op.Process();
    }
}
