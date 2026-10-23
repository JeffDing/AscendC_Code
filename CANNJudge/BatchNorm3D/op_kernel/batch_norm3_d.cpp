/*!
 * \file batch_norm3_d.cpp
 * \brief BatchNorm3D 算子 kernel 入口
 */

#include "batch_norm3_d.h"

enum class BatchNorm3DTilingKey : uint32_t
{
    TILING_KEY_BATCHNORM3D_MODE_0 = 0,
    TILING_KEY_BATCHNORM3D_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void batch_norm3_d(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean, GM_ADDR save_variance, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BatchNorm3DTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchNorm3DTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(BatchNorm3DTilingKey::TILING_KEY_BATCHNORM3D_MODE_0)) {
        NsBatchNorm3D::BatchNorm3D<half> op;
        op.Init(x, scale, offset, y, save_mean, save_variance, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(BatchNorm3DTilingKey::TILING_KEY_BATCHNORM3D_MODE_1)) {
        NsBatchNorm3D::BatchNorm3D<float> op;
        op.Init(x, scale, offset, y, save_mean, save_variance, &tilingData);
        op.Process();
    }
}
