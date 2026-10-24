/*!
 * \file mse_loss.h
 * \brief MseLoss 算子 kernel 类定义
 */

#ifndef MSELOSS_H
#define MSELOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mse_loss_tiling_data.h"
#include "mse_loss_tiling_key.h"

namespace NsMseLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class MseLoss {
public:
    __aicore__ inline MseLoss(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR target, GM_ADDR out, const MseLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueT;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<TPosition::VECCALC> diffBuf_;
    TBuf<TPosition::VECCALC> sqBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> outBuf_;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMT;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t reduction_ = 1;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 1;
    int64_t offset_ = 0;
    int64_t myCount_ = 0;
    float acc_ = 0.0f;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void MseLoss<T>::Init(GM_ADDR self, GM_ADDR target, GM_ADDR out, const MseLossTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    if (blockLength_ < 1) {
        blockLength_ = 1;
    }
    // ubFactor 字段高 16 位编码 reduction(none→1, mean→0, sum→2), 低 48 位为每轮 UB 元素数
    int64_t reductionCode = tilingData->ubFactor >> 48;
    ubLength_ = tilingData->ubFactor & ((1LL << 48) - 1);
    if (reductionCode == 1) {
        reduction_ = 0;
    } else if (reductionCode == 2) {
        reduction_ = 2;
    } else {
        reduction_ = 1;
    }
    if (ubLength_ < 1) {
        ubLength_ = 1;
    }

    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    blockNum_ = static_cast<int64_t>(GetBlockNum());
    offset_ = blockIdx_ * blockLength_;
    myCount_ = totalNum_ - offset_;
    if (myCount_ > blockLength_) {
        myCount_ = blockLength_;
    }
    if (myCount_ < 0) {
        myCount_ = 0;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
    inputGMT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(target));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    const uint32_t elemBytes = static_cast<uint32_t>(sizeof(T));
    const uint32_t slotBytes = static_cast<uint32_t>(ubLength_ * elemBytes);
    const uint32_t floatBytes = static_cast<uint32_t>(ubLength_ * 4);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, slotBytes);
    pipe.InitBuffer(inputQueueT, BUFFER_NUM, slotBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, slotBytes);
    pipe.InitBuffer(diffBuf_, floatBytes);
    pipe.InitBuffer(sqBuf_, floatBytes);
    pipe.InitBuffer(redBuf_, 256);
    pipe.InitBuffer(tmpBuf_, 256);
    pipe.InitBuffer(outBuf_, 128);

    // 归约模式(mean/sum): float 多核 AtomicAdd 累加, block0 先清零 output[0] 再 SyncAll
    // half 单核直写(覆盖), 不需要清零
    if (reduction_ != 0) {
        if constexpr (sizeof(T) == 4) {
            if (blockIdx_ == 0) {
                auto outLocal = outBuf_.Get<T>();
                Duplicate(outLocal, static_cast<T>(0), 1);
                DataCopyExtParams zeroParams{1, elemBytes, 0, 0, 0};
                DataCopyPad(outputGMY, outLocal, zeroParams);
            }
            SyncAll();
        }
    }
}

template <typename T>
__aicore__ inline void MseLoss<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> x = inputQueueX.AllocTensor<T>();
    LocalTensor<T> t = inputQueueT.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(x, inputGMX[offset_ + progress], copyParams, padParams);
    DataCopyPad(t, inputGMT[offset_ + progress], copyParams, padParams);
    inputQueueX.EnQue<T>(x);
    inputQueueT.EnQue<T>(t);
}

template <typename T>
__aicore__ inline void MseLoss<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> x = inputQueueX.DeQue<T>();
    LocalTensor<T> t = inputQueueT.DeQue<T>();

    // count 直接用 currentNum: AscendC Sub/Mul/Cast/ReduceSum Level 2 接口内部按 mask 处理非对齐
    const int32_t cnt = static_cast<int32_t>(currentNum);

    if (reduction_ == 0) {
        // 逐元素模式: y = (x - t)²
        LocalTensor<T> y = outputQueueY.AllocTensor<T>();
        if constexpr (sizeof(T) == 2) {
            // half: cast float32 计算 (x-y)², 再 cast half(与 golden float32 中间计算对齐)
            auto diffF = diffBuf_.Get<float>();
            auto sqF = sqBuf_.Get<float>();
            Cast(diffF, x, RoundMode::CAST_NONE, cnt);
            Cast(sqF, t, RoundMode::CAST_NONE, cnt);
            Sub(diffF, diffF, sqF, cnt);
            Mul(diffF, diffF, diffF, cnt);
            Cast(y, diffF, RoundMode::CAST_NONE, cnt);
        } else {
            Sub(y, x, t, cnt);
            Mul(y, y, y, cnt);
        }
        inputQueueX.FreeTensor(x);
        inputQueueT.FreeTensor(t);
        outputQueueY.EnQue<T>(y);
    } else {
        // 归约模式: 累加 (x - t)² 到 acc_(fp32 精确累加)
        auto diffF = diffBuf_.Get<float>();
        auto redLocal = redBuf_.Get<float>();
        auto tmpLocal = tmpBuf_.Get<float>();
        if constexpr (sizeof(T) == 2) {
            auto sqF = sqBuf_.Get<float>();
            Cast(diffF, x, RoundMode::CAST_NONE, cnt);
            Cast(sqF, t, RoundMode::CAST_NONE, cnt);
            Sub(diffF, diffF, sqF, cnt);
            Mul(diffF, diffF, diffF, cnt);
        } else {
            Sub(diffF, x, t, cnt);
            Mul(diffF, diffF, diffF, cnt);
        }
        ReduceSum(redLocal, diffF, tmpLocal, cnt);
        acc_ += redLocal.GetValue(0);
        inputQueueX.FreeTensor(x);
        inputQueueT.FreeTensor(t);
    }
}

template <typename T>
__aicore__ inline void MseLoss<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    if (reduction_ == 0) {
        // 逐元素模式: 写出 currentNum 个结果到 output[offset_ + progress]
        LocalTensor<T> y = outputQueueY.DeQue<T>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGMY[offset_ + progress], y, outParams);
        outputQueueY.FreeTensor(y);
    } else {
        // 归约模式: 写标量到 output[0]; mean 模式除以 N, sum 模式直接累加
        float finalVal = acc_;
        if (reduction_ == 1 && totalNum_ > 0) {
            finalVal = acc_ / static_cast<float>(totalNum_);
        }
        auto outLocal = outBuf_.Get<T>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        if constexpr (sizeof(T) == 4) {
            // float: 多核 fp32 原子累加, 累计舍入误差 ~N_cores * 2^-24, 远小于 1e-4 精度要求
            Duplicate(outLocal, finalVal, 1);
            SetAtomicAdd<T>();
            DataCopyPad(outputGMY, outLocal, outParams);
            DisableDmaAtomic();
        } else if (blockNum_ == 1) {
            // half: host 固定单核调度, 核内 fp32 精确累加, 仅输出一次 half 舍入
            Duplicate(outLocal, static_cast<T>(finalVal), 1);
            DataCopyPad(outputGMY, outLocal, outParams);
        } else {
            // half 意外多核兜底(正常调度不进入): half 原子累加
            Duplicate(outLocal, static_cast<T>(finalVal), 1);
            SetAtomicAdd<T>();
            DataCopyPad(outputGMY, outLocal, outParams);
            DisableDmaAtomic();
        }
    }
}

template <typename T>
__aicore__ inline void MseLoss<T>::Process()
{
    // TODO: 实现 Process 逻辑
    for (int64_t r = 0; r < myCount_; r += ubLength_) {
        int64_t cur = myCount_ - r;
        if (cur > ubLength_) {
            cur = ubLength_;
        }
        CopyIn(r, cur);
        if (reduction_ == 0) {
            Compute(cur);
            CopyOut(r, cur);
        } else {
            Compute(cur);
        }
    }
    if (reduction_ != 0) {
        CopyOut(0, 1);
    }
}

} // namespace NsMseLoss
#endif // MSELOSS_H
