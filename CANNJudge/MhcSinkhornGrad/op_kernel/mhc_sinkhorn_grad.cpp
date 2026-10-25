/*!
 * \file mhc_sinkhorn_grad.cpp
 * \brief MhcSinkhornGrad 算子 kernel 入口
 */

#include "mhc_sinkhorn_grad.h"

enum class MhcSinkhornGradTilingKey : uint32_t
{
    TILING_KEY_MHCSINKHORNGRAD_MODE_0 = 0,
    TILING_KEY_MHCSINKHORNGRAD_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void mhc_sinkhorn_grad(GM_ADDR gradOutput, GM_ADDR normOut, GM_ADDR sumOut, GM_ADDR gradInput, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcSinkhornGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornGradTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(MhcSinkhornGradTilingKey::TILING_KEY_MHCSINKHORNGRAD_MODE_0)) {
        NsMhcSinkhornGrad::MhcSinkhornGrad<half> op;
        op.Init(gradOutput, normOut, sumOut, gradInput, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(MhcSinkhornGradTilingKey::TILING_KEY_MHCSINKHORNGRAD_MODE_1)) {
        NsMhcSinkhornGrad::MhcSinkhornGrad<float> op;
        op.Init(gradOutput, normOut, sumOut, gradInput, &tilingData);
        op.Process();
    }
}
