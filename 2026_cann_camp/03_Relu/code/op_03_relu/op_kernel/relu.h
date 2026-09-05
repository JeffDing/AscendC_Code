/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"
#include <algorithm>

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
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
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    int64_t ubFactor = tilingData->ubFactor;

    int64_t blockIdx = GetBlockIdx();
    int64_t blockDim = GetBlockNum();

    int64_t offset = blockIdx * blockFactor;
    int64_t length = blockFactor;
    if (blockIdx == blockDim - 1) {
        length = totalNum - offset;
        if (length < 0) {
            length = 0;
        }
    }
    if (length <= 0) {
        blockLength_ = 0;
        ubLength_ = ubFactor;
        return;
    }

    blockLength_ = length;
    ubLength_ = ubFactor;

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubFactor * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubFactor * sizeof(T));

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, length);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, length);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t length = blockLength_;
    int64_t ubFactor = ubLength_;

    for (int64_t progress = 0; progress < length; progress += ubFactor) {
        int64_t currentNum = (ubFactor < length - progress) ? ubFactor : (length - progress);

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H