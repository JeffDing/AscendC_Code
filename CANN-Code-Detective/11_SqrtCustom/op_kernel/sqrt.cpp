/*!
 * \file sqrt.cpp
 * \brief Sqrt 算子 kernel 入口
 */

#include "sqrt.h"

enum class SqrtTilingKey : uint32_t
{
    TILING_KEY_SQRT_MODE_0 = 0,
    TILING_KEY_SQRT_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SqrtTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SqrtTilingKey::TILING_KEY_SQRT_MODE_0)) {
        NsSqrt::Sqrt<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SqrtTilingKey::TILING_KEY_SQRT_MODE_1)) {
        NsSqrt::Sqrt<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
