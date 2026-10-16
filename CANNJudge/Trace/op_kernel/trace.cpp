/*!
 * \file trace.cpp
 * \brief Trace 算子 kernel 入口
 */

#include "trace.h"

enum class TraceTilingKey : uint32_t
{
    TILING_KEY_TRACE_MODE_0 = 0,
    TILING_KEY_TRACE_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void trace(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TraceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TraceTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(TraceTilingKey::TILING_KEY_TRACE_MODE_0)) {
        NsTrace::Trace<half> op;
        op.Init(self, out, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(TraceTilingKey::TILING_KEY_TRACE_MODE_1)) {
        NsTrace::Trace<float> op;
        op.Init(self, out, &tilingData);
        op.Process();
    }
}
