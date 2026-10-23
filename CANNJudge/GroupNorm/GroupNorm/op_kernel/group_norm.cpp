/*!
 * \file group_norm.cpp
 * \brief GroupNorm 算子 kernel 入口
 */

#include "group_norm.h"

enum class GroupNormTilingKey : uint32_t
{
    TILING_KEY_GROUPNORM_MODE_0 = 0,
    TILING_KEY_GROUPNORM_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void group_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GroupNormTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupNormTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(GroupNormTilingKey::TILING_KEY_GROUPNORM_MODE_0)) {
        NsGroupNorm::GroupNorm<half> op;
        op.Init(x, gamma, beta, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormTilingKey::TILING_KEY_GROUPNORM_MODE_1)) {
        NsGroupNorm::GroupNorm<float> op;
        op.Init(x, gamma, beta, y, &tilingData);
        op.Process();
    }
}
