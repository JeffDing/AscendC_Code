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
    int64_t totalNum_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;
    this->totalNum_ = tilingData->totalNum;

    uint32_t blockId = GetBlockIdx();
    int64_t gmOffset = static_cast<int64_t>(blockId) * this->blockLength_;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset);

    int64_t bufLen = (this->ubLength_ < this->blockLength_) ? this->ubLength_ : this->blockLength_;
    if (bufLen <= 0) {
        bufLen = this->ubLength_;
    }
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufLen * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufLen * static_cast<int64_t>(sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    Maxs(yLocal, xLocal, static_cast<T>(0), currentNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    uint32_t blockId = GetBlockIdx();
    int64_t gmOffset = static_cast<int64_t>(blockId) * this->blockLength_;
    int64_t myCount = this->totalNum_ - gmOffset;
    if (myCount > this->blockLength_) {
        myCount = this->blockLength_;
    }
    if (myCount <= 0) {
        return;
    }

    int64_t tileNum = (this->ubLength_ > 0) ? (myCount / this->ubLength_) : 0;
    int64_t remainNum = myCount - tileNum * this->ubLength_;

    for (int64_t i = 0; i < tileNum; i++) {
        CopyIn(i, this->ubLength_);
        Compute(this->ubLength_);
        CopyOut(i, this->ubLength_);
    }
    if (remainNum > 0) {
        CopyIn(tileNum, remainNum);
        Compute(remainNum);
        CopyOut(tileNum, remainNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
