/*!
 * \file trace.h
 * \brief Trace 算子 kernel 类定义
 */

#ifndef TRACE_H
#define TRACE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "trace_tiling_data.h"
#include "trace_tiling_key.h"

namespace NsTrace {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 每轮处理的元素数上限: UB 预算(实测 192KB 池, fp32=128F+896 / fp16=192F+640)与
// convBuf 单块 64KB、Duplicate mask 版 repeatTime(uint8, F/8<=255) 的共同安全值
constexpr int64_t TRACE_UB_FACTOR_CAP = 1008;

template <typename T>
class Trace {
public:
    __aicore__ inline Trace(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const TraceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void BuildMask();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> convBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> outBuf_;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t strideN_ = 0;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 1;
    int64_t offset_ = 0;
    int64_t myCount_ = 0;
    float acc_ = 0.0f;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Trace<T>::Init(GM_ADDR self, GM_ADDR out, const TraceTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    // ubFactor 字段承载矩阵列数 N(对角线相邻元素间隔 N+1, kernel 侧必需信息)
    strideN_ = tilingData->ubFactor;
    if (strideN_ < 1) {
        strideN_ = 1;
    }
    // 每轮元素数: 不超过每核元素数与 UB 容量安全上限, 核数充足时每核单轮完成
    ubLength_ = (blockLength_ < TRACE_UB_FACTOR_CAP) ? blockLength_ : TRACE_UB_FACTOR_CAP;
    if (ubLength_ < 1) {
        ubLength_ = 1;
    }

    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    offset_ = blockIdx_ * blockLength_;
    myCount_ = totalNum_ - offset_;
    if (myCount_ > blockLength_) {
        myCount_ = blockLength_;
    }
    if (myCount_ < 0) {
        myCount_ = 0;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    const uint32_t slotBytes = static_cast<uint32_t>(ubLength_ * 32);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, slotBytes);
    pipe.InitBuffer(maskBuf_, slotBytes);
    pipe.InitBuffer(yBuf_, slotBytes);
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(convBuf_, static_cast<uint32_t>(ubLength_ * 64));
    } else {
        pipe.InitBuffer(convBuf_, 256);
    }
    pipe.InitBuffer(redBuf_, 256);
    pipe.InitBuffer(tmpBuf_, 256);
    pipe.InitBuffer(outBuf_, 128);

    BuildMask();

    auto outLocal = outBuf_.Get<T>();
    if (blockIdx_ == 0) {
        DataCopyExtParams zeroParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        Duplicate(outLocal, static_cast<T>(0), 1);
        DataCopyPad(outputGMY, outLocal, zeroParams);
    }
    SyncAll();
}

// 构造 gather 掩码: 每个 32B 槽内仅 lane0(对角线元素)参与计算
template <typename T>
__aicore__ inline void Trace<T>::BuildMask()
{
    constexpr int64_t slotElems = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t factor = ubLength_;
    auto maskLocal = maskBuf_.Get<T>();
    uint64_t maskArr[2];
    if constexpr (sizeof(T) == 4) {
        maskArr[0] = 0x0101010101010101ULL;
        maskArr[1] = 0ULL;
    } else {
        maskArr[0] = 0x0001000100010001ULL;
        maskArr[1] = 0x0001000100010001ULL;
    }
    Duplicate(maskLocal, static_cast<T>(0), static_cast<int32_t>(factor * slotElems));
    const int64_t fullReps = factor / 8;
    const int64_t tailSlots = factor % 8;
    if (fullReps > 0) {
        Duplicate(maskLocal, static_cast<T>(1), maskArr, static_cast<uint8_t>(fullReps), 1, 8);
    }
    if (tailSlots > 0) {
        uint64_t tailArr[2] = {0ULL, 0ULL};
        if constexpr (sizeof(T) == 4) {
            tailArr[0] = ((1ULL << (8 * tailSlots)) - 1ULL) / 0xFFULL;
        } else {
            for (uint32_t s = 0; s < static_cast<uint32_t>(tailSlots); ++s) {
                if (s < 4) {
                    tailArr[0] |= 1ULL << (16 * s);
                } else {
                    tailArr[1] |= 1ULL << (16 * (s - 4));
                }
            }
        }
        Duplicate(maskLocal[static_cast<int32_t>(fullReps * 256 / sizeof(T))], static_cast<T>(1), tailArr, 1, 1, 8);
    }
}

template <typename T>
__aicore__ inline void Trace<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> x = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(currentNum), static_cast<uint32_t>(sizeof(T)),
                                 static_cast<uint32_t>(strideN_ * static_cast<int64_t>(sizeof(T))), 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(x, inputGMX[(offset_ + progress) * (strideN_ + 1)], copyParams, padParams);
    inputQueueX.EnQue<T>(x);
}

template <typename T>
__aicore__ inline void Trace<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xd = inputQueueX.DeQue<T>();
    constexpr int64_t slotElems = 32 / static_cast<int64_t>(sizeof(T));
    const int32_t cnt = static_cast<int32_t>(currentNum * slotElems);
    auto yLocal = yBuf_.Get<T>();
    auto redLocal = redBuf_.Get<float>();
    auto tmpLocal = tmpBuf_.Get<float>();
    Mul(yLocal, xd, maskBuf_.Get<T>(), cnt);
    if constexpr (sizeof(T) == 2) {
        auto convLocal = convBuf_.Get<float>();
        Cast(convLocal, yLocal, RoundMode::CAST_NONE, cnt);
        ReduceSum(redLocal, convLocal, tmpLocal, cnt);
    } else {
        ReduceSum(redLocal, yLocal, tmpLocal, cnt);
    }
    acc_ += redLocal.GetValue(0);
    inputQueueX.FreeTensor(xd);
}

template <typename T>
__aicore__ inline void Trace<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    (void)progress;
    (void)currentNum;
    auto outLocal = outBuf_.Get<T>();
    DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    if constexpr (sizeof(T) == 4) {
        // fp32: 各核 fp32 原子累加, 累计舍入误差 ~2^-24, 远小于 2^-13 精度要求
        Duplicate(outLocal, acc_, 1);
        SetAtomicAdd<T>();
        DataCopyPad(outputGMY, outLocal, outParams);
        DisableDmaAtomic();
    } else if (blockNum_ == 1) {
        // fp16: host 侧固定单核调度, 核内 fp32 精确累加, 仅输出一次 half 舍入
        Duplicate(outLocal, static_cast<T>(acc_), 1);
        DataCopyPad(outputGMY, outLocal, outParams);
    } else {
        // fp16 意外多核时的兜底路径(正常调度不会进入): half 原子累加
        Duplicate(outLocal, static_cast<T>(acc_), 1);
        SetAtomicAdd<T>();
        DataCopyPad(outputGMY, outLocal, outParams);
        DisableDmaAtomic();
    }
}

template <typename T>
__aicore__ inline void Trace<T>::Process()
{
    // TODO: 实现 Process 逻辑
    for (int64_t r = 0; r < myCount_; r += ubLength_) {
        int64_t cur = myCount_ - r;
        if (cur > ubLength_) {
            cur = ubLength_;
        }
        CopyIn(r, cur);
        Compute(cur);
    }
    CopyOut(0, 1);
}

} // namespace NsTrace
#endif // TRACE_H
