/*!
 * \file quant_matmul_sigmoid_mul_quant.cpp
 * \brief QuantMatmulSigmoidMulQuant 算子 kernel 入口
 */

#include "quant_matmul_sigmoid_mul_quant.h"

enum class QuantMatmulSigmoidMulQuantTilingKey : uint32_t
{
    TILING_KEY_QUANTMATMULSIGMOIDMULQUANT_MODE_0 = 0,
    TILING_KEY_QUANTMATMULSIGMOIDMULQUANT_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void quant_matmul_sigmoid_mul_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR bias, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantMatmulSigmoidMulQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulSigmoidMulQuantTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(QuantMatmulSigmoidMulQuantTilingKey::TILING_KEY_QUANTMATMULSIGMOIDMULQUANT_MODE_0)) {
        NsQuantMatmulSigmoidMulQuant::QuantMatmulSigmoidMulQuant<half> op;
        op.Init(x1, x2, x1Scale, x2Scale, bias, y, yScale, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantMatmulSigmoidMulQuantTilingKey::TILING_KEY_QUANTMATMULSIGMOIDMULQUANT_MODE_1)) {
        NsQuantMatmulSigmoidMulQuant::QuantMatmulSigmoidMulQuant<int8_t> op;
        op.Init(x1, x2, x1Scale, x2Scale, bias, y, yScale, &tilingData);
        op.Process();
    }
}
