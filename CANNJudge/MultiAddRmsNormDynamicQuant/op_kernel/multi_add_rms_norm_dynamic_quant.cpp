/*!
 * \file multi_add_rms_norm_dynamic_quant.cpp
 * \brief MultiAddRmsNormDynamicQuant 算子 kernel 入口
 */

#include "multi_add_rms_norm_dynamic_quant.h"

enum class MultiAddRmsNormDynamicQuantTilingKey : uint32_t
{
    TILING_KEY_MULTIADDRMSNORMDYNAMICQUANT_MODE_0 = 0,
    TILING_KEY_MULTIADDRMSNORMDYNAMICQUANT_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void multi_add_rms_norm_dynamic_quant(GM_ADDR x1a, GM_ADDR x1b, GM_ADDR x1c, GM_ADDR x1d, GM_ADDR x1e, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smoothScale1, GM_ADDR smoothScale2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MultiAddRmsNormDynamicQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(MultiAddRmsNormDynamicQuantTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(MultiAddRmsNormDynamicQuantTilingKey::TILING_KEY_MULTIADDRMSNORMDYNAMICQUANT_MODE_0)) {
        NsMultiAddRmsNormDynamicQuant::MultiAddRmsNormDynamicQuant<half> op;
        op.Init(x1a, x1b, x1c, x1d, x1e, x2, gamma, smoothScale1, smoothScale2, y1, y2, x, y, scale1, scale2, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(MultiAddRmsNormDynamicQuantTilingKey::TILING_KEY_MULTIADDRMSNORMDYNAMICQUANT_MODE_1)) {
        NsMultiAddRmsNormDynamicQuant::MultiAddRmsNormDynamicQuant<float> op;
        op.Init(x1a, x1b, x1c, x1d, x1e, x2, gamma, smoothScale1, smoothScale2, y1, y2, x, y, scale1, scale2, &tilingData);
        op.Process();
    }
}
