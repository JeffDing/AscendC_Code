/*!
 * \file group_norm_silu_quant.cpp
 * \brief GroupNormSiluQuant 算子 kernel 入口
 */

#include "group_norm_silu_quant.h"

enum class GroupNormSiluQuantTilingKey : uint32_t
{
    TILING_KEY_GROUPNORMSILUQUANT_MODE_0 = 0,
    TILING_KEY_GROUPNORMSILUQUANT_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void group_norm_silu_quant(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GroupNormSiluQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupNormSiluQuantTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_0)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<half> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GroupNormSiluQuantTilingKey::TILING_KEY_GROUPNORMSILUQUANT_MODE_1)) {
        NsGroupNormSiluQuant::GroupNormSiluQuant<float> op;
        op.Init(self, gamma, beta, quantScale, out, meanOut, rstdOut, &tilingData);
        op.Process();
    }
}
