/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

// 每轮 UB 处理元素数上限: 双队列(VECIN/VECOUT)各 2 深度, 合计 4*cap*sizeof(T) 字节,
// 须 < 192KB UB 池; cap=2048 时 fp32=32KB/fp16=16KB, 留有充足余量
constexpr int64_t ABS_UB_FACTOR_CAP = 2048;

template <typename T>
class Abs {
public:
    __aicore__ inline Abs() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 1;
    int64_t offset_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Abs<T>::Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (blockLength_ < 1) {
        blockLength_ = 1;
    }
    if (ubLength_ < 1) {
        ubLength_ = 1;
    }
    // 每轮 UB 元素数受 192KB UB 池上限约束(UT 占位 ubFactor 可能远超, 须在此收敛)
    if (ubLength_ > ABS_UB_FACTOR_CAP) {
        ubLength_ = ABS_UB_FACTOR_CAP;
    }
    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum_);
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum_);
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    blockNum_ = static_cast<int64_t>(GetBlockNum());
    offset_ = blockIdx_ * blockLength_;

    // TPipe 必须显式 InitBuffer, 否则 TQue 的 bufNum 为 0, AllocTensor 会 abort
    const uint32_t slotBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe_.InitBuffer(inputQueue_, 2, slotBytes);
    pipe_.InitBuffer(outputQueue_, 2, slotBytes);
}

// TODO: 实现 CopyIn 逻辑
template <typename T>
__aicore__ inline void Abs<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> in = inputQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(in, inputGM_[offset_ + progress], copyParams, padParams);
    inputQueue_.EnQue<T>(in);
}

// TODO: 实现 Compute 逻辑
template <typename T>
__aicore__ inline void Abs<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> in = inputQueue_.DeQue<T>();
    LocalTensor<T> out = outputQueue_.AllocTensor<T>();
    // 显式限定命名空间, 避免与注入类名 Abs<T> 歧义
    AscendC::Abs(out, in, static_cast<int32_t>(currentNum));
    inputQueue_.FreeTensor(in);
    outputQueue_.EnQue<T>(out);
}

// TODO: 实现 CopyOut 逻辑
template <typename T>
__aicore__ inline void Abs<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> out = outputQueue_.DeQue<T>();
    DataCopyExtParams outParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGM_[offset_ + progress], out, outParams);
    outputQueue_.FreeTensor(out);
}

template <typename T>
__aicore__ inline void Abs<T>::Process()
{
    // TODO: 实现 Process 逻辑
    int64_t total = totalNum_;
    for (int64_t i = 0; i < blockLength_; i += ubLength_) {
        int64_t cur = ubLength_;
        int64_t offset = offset_ + i;
        if (offset >= total) {
            break;
        }
        if (offset + cur > total) {
            cur = total - offset;
        }
        CopyIn(i, cur);
        Compute(cur);
        CopyOut(i, cur);
    }
}

} // namespace NsAbs

#endif // ABS_H
