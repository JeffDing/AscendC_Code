/*!
 * \file abs.cpp
 * \brief Abs 算子 kernel 入口
 */

#include "abs.h"

enum class AbsTilingKey : uint32_t
{
    TILING_KEY_ABS_MODE_0 = 0,
    TILING_KEY_ABS_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AbsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AbsTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(AbsTilingKey::TILING_KEY_ABS_MODE_0)) {
        NsAbs::Abs<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(AbsTilingKey::TILING_KEY_ABS_MODE_1)) {
        NsAbs::Abs<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
