/*!
 * \file multi_add_rms_norm_dynamic_quant_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _MULTIADDRMSNORMDYNAMICQUANT_TILING_DATA_H_
#define _MULTIADDRMSNORMDYNAMICQUANT_TILING_DATA_H_

struct MultiAddRmsNormDynamicQuantTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};
#endif
