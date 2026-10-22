/*!
 * \file bn_training_update.cpp
 * \brief BnTrainingUpdate 算子 kernel 入口
 */

#include "bn_training_update.h"

enum class BnTrainingUpdateTilingKey : uint32_t
{
    TILING_KEY_BNTRAININGUPDATE_MODE_0 = 0,
    TILING_KEY_BNTRAININGUPDATE_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void bn_training_update(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean, GM_ADDR save_variance, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BnTrainingUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(BnTrainingUpdateTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(BnTrainingUpdateTilingKey::TILING_KEY_BNTRAININGUPDATE_MODE_0)) {
        NsBnTrainingUpdate::BnTrainingUpdate<half> op;
        op.Init(x, sum, square_sum, scale, offset, y, save_mean, save_variance, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(BnTrainingUpdateTilingKey::TILING_KEY_BNTRAININGUPDATE_MODE_1)) {
        NsBnTrainingUpdate::BnTrainingUpdate<float> op;
        op.Init(x, sum, square_sum, scale, offset, y, save_mean, save_variance, &tilingData);
        op.Process();
    }
}
