/*!
 * \file sqrt.h
 * \brief Sqrt 算子 kernel 类定义
 */

// CANN 9.0.0 内部头 <asc/impl/utils/std/cmath/sqrt.h> 在部分工具链的包含链里会
// #include "sqrt.h"，而 opc 预编译把源码目录置于 -I 前端，导致递归包含自身、
// 最终报 #include nested too deeply。此处提前置灰该头 guard 使其退化为空壳；
// 本 kernel 使用向量级 AscendC::Sqrt(yLocal, xLocal, count)，
// 不依赖 AscendC::Std::sqrt，压制该头不影响功能。
#ifndef IMPL_STD_SQRT_H
#define IMPL_STD_SQRT_H
#endif

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// UB 预算兜底：910B UB 共 256KB，系统预留 1/4 后可用 3/4；
// 输入/输出两条队列，每队列 BUFFER_NUM 个 buffer
constexpr int64_t UB_TOTAL_BYTES = 262144;
constexpr int64_t UB_USABLE_NUM = 3;
constexpr int64_t UB_USABLE_DEN = 4;
constexpr int64_t UB_QUEUE_NUM = 2;

// NPU 上 MTE 搬运要求 GM/UB 地址及搬运粒度按 32 字节对齐，
// 非对齐尾块需走 DataCopyPad
constexpr int64_t ALIGN_BYTES = 32;

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
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

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Sqrt<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    const int64_t blockIdx = GetBlockIdx();
    const int64_t blockOffset = blockIdx * tilingData->blockFactor;
    blockLength_ = tilingData->totalNum - blockOffset;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    const int64_t alignElems = ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t ubLimit = UB_TOTAL_BYTES * UB_USABLE_NUM / UB_USABLE_DEN /
        (static_cast<int64_t>(sizeof(T)) * UB_QUEUE_NUM * BUFFER_NUM);
    // ubFactor 需按 32B 粒度对齐并钳制到 UB 预算内，保证核内循环的 GM 偏移始终对齐
    int64_t ubFactor = tilingData->ubFactor;
    if (ubFactor > ubLimit) {
        ubFactor = ubLimit;
    }
    ubFactor = ubFactor / alignElems * alignElems;
    if (ubFactor < alignElems) {
        ubFactor = alignElems;
    }
    ubLength_ = (blockLength_ < ubFactor) ? blockLength_ : ubFactor;

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const GlobalTensor<T>& src = inputGMX[static_cast<uint64_t>(progress * ubLength_)];
    if ((currentNum * static_cast<int64_t>(sizeof(T))) % ALIGN_BYTES == 0) {
        DataCopy(xLocal, src, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyPad(xLocal, src, DataCopyExtParams(1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0),
            DataCopyPadExtParams<T>());
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const int32_t sqrtCount = static_cast<int32_t>(currentNum);
    AscendC::Sqrt(yLocal, xLocal, sqrtCount);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    GlobalTensor<T> dst = outputGMY[static_cast<uint64_t>(progress * ubLength_)];
    if ((currentNum * static_cast<int64_t>(sizeof(T))) % ALIGN_BYTES == 0) {
        DataCopy(dst, yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyPad(dst, yLocal, DataCopyExtParams(1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0));
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t progress = 0;
    int64_t remainNum = blockLength_;
    while (remainNum > 0) {
        const int64_t currentNum = (remainNum < ubLength_) ? remainNum : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
        progress++;
        remainNum -= currentNum;
    }
}

} // namespace NsSqrt
#endif // SQRT_H