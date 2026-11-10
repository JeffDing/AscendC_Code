/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    int64_t remainder = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainder > tilingData->blockFactor) ? tilingData->blockFactor : remainder;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * blockIdx, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * blockIdx, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto localX = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(localX, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(localX);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    auto localX = inputQueueX.DeQue<T>();
    auto localY = outputQueueY.AllocTensor<T>();

    AscendC::Sigmoid(localY, localX, static_cast<uint32_t>(currentNum));

    inputQueueX.FreeTensor(localX);
    outputQueueY.EnQue(localY);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto localY = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], localY, copyParams);
    outputQueueY.FreeTensor(localY);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t totalProgress = CeilDiv(blockLength_, ubLength_);
    for (int64_t progress = 0; progress < totalProgress; progress++) {
        int64_t currentNum = (progress == totalProgress - 1)
            ? (blockLength_ - progress * ubLength_)
            : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSigmoid
#endif // SIGMOID_H
