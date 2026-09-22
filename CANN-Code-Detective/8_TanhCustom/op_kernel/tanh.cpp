/*!
 * \file tanh.cpp
 * \brief Tanh 算子 kernel 入口
 */

#include "tanh.h"

enum class TanhTilingKey : uint32_t
{
    TILING_KEY_TANH_MODE_0 = 0,
    TILING_KEY_TANH_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(TanhTilingKey::TILING_KEY_TANH_MODE_0)) {
        NsTanh::Tanh<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(TanhTilingKey::TILING_KEY_TANH_MODE_1)) {
        NsTanh::Tanh<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
