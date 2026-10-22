/*!
 * \file flash_attention.cpp
 * \brief FlashAttention 算子 kernel 入口
 */

#include "flash_attention.h"

enum class FlashAttentionTilingKey : uint32_t
{
    TILING_KEY_FLASHATTENTION_MODE_0 = 0,
    TILING_KEY_FLASHATTENTION_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void flash_attention(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(FlashAttentionTilingKey::TILING_KEY_FLASHATTENTION_MODE_0)) {
        NsFlashAttention::FlashAttention<half> op;
        op.Init(q, k, v, attn_mask, out, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(FlashAttentionTilingKey::TILING_KEY_FLASHATTENTION_MODE_1)) {
        NsFlashAttention::FlashAttention<float> op;
        op.Init(q, k, v, attn_mask, out, &tilingData);
        op.Process();
    }
}
