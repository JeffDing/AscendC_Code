/*!
 * \file leakyrelu.cpp
 * \brief Leakyrelu 算子 kernel 入口
 */

#include "leakyrelu.h"

enum class LeakyreluTilingKey : uint32_t
{
    TILING_KEY_LEAKYRELU_MODE_0 = 0,
    TILING_KEY_LEAKYRELU_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void leakyrelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LeakyreluTilingData);
    GET_TILING_DATA_WITH_STRUCT(LeakyreluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(LeakyreluTilingKey::TILING_KEY_LEAKYRELU_MODE_0)) {
        NsLeakyrelu::Leakyrelu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(LeakyreluTilingKey::TILING_KEY_LEAKYRELU_MODE_1)) {
        NsLeakyrelu::Leakyrelu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
