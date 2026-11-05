/*!
 * \file depthwise_conv2_d_tiling.h
 * \brief DepthwiseConv2D tiling 数据定义
 */

#ifndef _I_DEPTHWISECONV2D_TILING_H_
#define _I_DEPTHWISECONV2D_TILING_H_

#include <cstdint>
#include <cstring>
#include "../../../op_kernel/depthwise_conv2_d_tiling_data.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "graph/c_types.h"
#include "ascendc/host_api/tiling/template_argument.h"

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __ubuf__
#define __ubuf__
#endif

inline void InitTilingData(uint8_t *tiling, DepthwiseConv2DTilingData *constData)
{
    memcpy(constData, tiling, sizeof(DepthwiseConv2DTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    DepthwiseConv2DTilingData tilingData;          \
    InitTilingData(tilingArg, &tilingData)

#endif
