/*!
 * \file quant_batch_mat_mul_v3.cpp
 * \brief QuantBatchMatMulV3 算子 kernel 入口
 */

#include "quant_batch_mat_mul_v3.h"

enum class QuantBatchMatMulV3TilingKey : uint32_t
{
    TILING_KEY_QUANTBATCHMATMULV3_MODE_0 = 0,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void quant_batch_mat_mul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantBatchMatMulV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantBatchMatMulV3TilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_0)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3<half> op;
        op.Init(x1, x2, scale, pertokenScale, out, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_1)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3<int8_t> op;
        op.Init(x1, x2, scale, pertokenScale, out, &tilingData);
        op.Process();
    }
}
