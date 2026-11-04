/*!
 * \file conv_transpose2_d.cpp
 * \brief ConvTranspose2D 算子 kernel 入口
 */

#include "conv_transpose2_d.h"

enum class ConvTranspose2DTilingKey : uint32_t
{
    TILING_KEY_CONVTRANSPOSE2D_MODE_0 = 0,
    TILING_KEY_CONVTRANSPOSE2D_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void conv_transpose2_d(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ConvTranspose2DTilingData);
    GET_TILING_DATA_WITH_STRUCT(ConvTranspose2DTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ConvTranspose2DTilingKey::TILING_KEY_CONVTRANSPOSE2D_MODE_0)) {
        NsConvTranspose2D::ConvTranspose2D<half> op;
        op.Init(x, w, b, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ConvTranspose2DTilingKey::TILING_KEY_CONVTRANSPOSE2D_MODE_1)) {
        NsConvTranspose2D::ConvTranspose2D<float> op;
        op.Init(x, w, b, y, &tilingData);
        op.Process();
    }
}
