/*!
 * \file mse_loss.cpp
 * \brief MseLoss 算子 kernel 入口
 */

#include "mse_loss.h"

enum class MseLossTilingKey : uint32_t
{
    TILING_KEY_MSELOSS_MODE_0 = 0,
    TILING_KEY_MSELOSS_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void mse_loss(GM_ADDR self, GM_ADDR target, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MseLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(MseLossTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(MseLossTilingKey::TILING_KEY_MSELOSS_MODE_0)) {
        NsMseLoss::MseLoss<half> op;
        op.Init(self, target, out, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(MseLossTilingKey::TILING_KEY_MSELOSS_MODE_1)) {
        NsMseLoss::MseLoss<float> op;
        op.Init(self, target, out, &tilingData);
        op.Process();
    }
}
