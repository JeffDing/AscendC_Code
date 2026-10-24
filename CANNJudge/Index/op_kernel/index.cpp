/*!
 * \file index.cpp
 * \brief Index 算子 kernel 入口
 */

#include "index.h"

enum class IndexTilingKey : uint32_t
{
    TILING_KEY_INDEX_MODE_0 = 0,
    TILING_KEY_INDEX_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void index(GM_ADDR x, GM_ADDR indexed_sizes, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(IndexTilingData);
    GET_TILING_DATA_WITH_STRUCT(IndexTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(IndexTilingKey::TILING_KEY_INDEX_MODE_0)) {
        NsIndex::Index<half> op;
        op.Init(x, indexed_sizes, indices, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(IndexTilingKey::TILING_KEY_INDEX_MODE_1)) {
        NsIndex::Index<float> op;
        op.Init(x, indexed_sizes, indices, y, &tilingData);
        op.Process();
    }
}
