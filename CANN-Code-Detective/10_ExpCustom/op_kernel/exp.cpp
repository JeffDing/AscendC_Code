/*!
 * \file exp.cpp
 * \brief Exp 算子 kernel 入口
 */

#include "exp.h"

enum class ExpTilingKey : uint32_t
{
    TILING_KEY_EXP_MODE_0 = 0,
    TILING_KEY_EXP_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ExpTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_MODE_0)) {
        NsExp::Exp<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_MODE_1)) {
        NsExp::Exp<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
