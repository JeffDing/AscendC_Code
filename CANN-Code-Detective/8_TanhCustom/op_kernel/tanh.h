/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 单次 UB 循环处理的安全字节数上限。
// 输入/输出各占 BUFFER_NUM 份 buffer，共需 2 * BUFFER_NUM * UB_CHUNK_BYTES；
// 取 32KB 时无论 UB 为 192KB 还是模拟器环境都不会溢出。
constexpr int64_t UB_CHUNK_BYTES = 32 * 1024;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
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
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 本核处理的数据总长度：按 blockFactor 均分，尾核不足时以剩余长度截断
    int64_t startOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = tilingData->blockFactor;
    int64_t remainNum = tilingData->totalNum - startOffset;
    if (remainNum < blockLength_) {
        blockLength_ = (remainNum > 0) ? remainNum : 0;
    }

    // 单次 UB 循环处理的元素数，限制在安全字节数内防止 UB 溢出
    int64_t maxUbElems = UB_CHUNK_BYTES / static_cast<int64_t>(sizeof(T));
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0 || ubLength_ > maxUbElems) {
        ubLength_ = maxUbElems;
    }

    // 设置本核的 Global Memory 起始地址
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // 为输入/输出队列分配 UB 内存（双缓冲）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // tanh(x) = 2 / (1 + exp(-2x)) - 1
    // 该形式对正负无穷均数值稳定：
    // x -> +inf 时 exp(-2x) -> 0，tanh 收敛到 1；
    // x -> -inf 时 exp(-2x) -> inf，2/inf -> 0，tanh 收敛到 -1
    Muls(yLocal, xLocal, static_cast<T>(-2.0), currentNum);
    Exp(yLocal, yLocal, currentNum);
    Adds(yLocal, yLocal, static_cast<T>(1.0), currentNum);

    // xLocal 数据已用完，复用为常量 2 的暂存，避免额外占用 UB
    Duplicate(xLocal, static_cast<T>(2.0), currentNum);
    Div(xLocal, xLocal, yLocal, currentNum);
    Adds(xLocal, xLocal, static_cast<T>(-1.0), currentNum);

    // 结果搬回输出队列
    DataCopy(yLocal, xLocal, currentNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // 循环处理本核数据，尾块按剩余元素数处理
    for (int64_t i = 0; i < blockLength_; i += ubLength_) {
        int64_t currentNum = blockLength_ - i;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
