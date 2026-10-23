/*!
 * \file batch_norm3_d.h
 * \brief BatchNorm3D 算子 kernel 类定义
 */

#ifndef BATCHNORM3D_H
#define BATCHNORM3D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm3_d_tiling_data.h"
#include "batch_norm3_d_tiling_key.h"

namespace NsBatchNorm3D {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 单块 UB 处理元素数（编译期固定，保证 UB 不溢出）
constexpr int64_t BN3D_UB_ELEMS = 4096;

// TilingData 位域约定（结构不可改，host / kernel 两侧严格一致）：
//   blockFactor : bit[0:39)  = DHW = D * H * W（每个 (n, c) 段的元素数）
//                 bit[39:52) = C（通道数）
//   ubFactor    : bit[0:32)  = epsilon 的 IEEE-754 位模式
//                 bit[32:64) = N（batch 数，即每通道的段数）
//   totalNum    : 保留（元素总数，kernel 由 N*C*DHW 推导，不依赖该字段）
constexpr uint64_t DHW_MASK = 0x7FFFFFFFFFULL;  // 39 bit
constexpr uint64_t C_MASK = 0x1FFFULL;          // 13 bit
constexpr uint64_t N_MASK = 0xFFFFFFFFULL;      // 32 bit

template <typename T>
class BatchNorm3D {
public:
    __aicore__ inline BatchNorm3D(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean, GM_ADDR save_variance, const BatchNorm3DTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Accumulate(int64_t currentNum);
    __aicore__ inline void Normalize(int64_t currentNum, float coefA, float coefB);
    __aicore__ inline float ReduceBuffer(const LocalTensor<float>& buf);
    __aicore__ inline void WriteChannelStats(int64_t localCount);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> floatBuf_;     // fp16 输入升 fp32 的中间缓冲
    TBuf<TPosition::VECCALC> sqBuf_;        // x * x 的中间缓冲
    TBuf<TPosition::VECCALC> accBuf_;       // sum 累加器
    TBuf<TPosition::VECCALC> accSqBuf_;     // sum of squares 累加器
    TBuf<TPosition::VECCALC> redBuf_;       // 归约结果
    TBuf<TPosition::VECCALC> tmpBuf_;       // 归约 sharedTmpBuffer
    TBuf<TPosition::VECCALC> statMeanBuf_;  // 本核各通道 mean（经 MTE3 写出）
    TBuf<TPosition::VECCALC> statVarBuf_;   // 本核各通道 var（经 MTE3 写出）

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> scaleGM;
    GlobalTensor<float> offsetGM;
    GlobalTensor<float> saveMeanGM;
    GlobalTensor<float> saveVarGM;

    int64_t n_ = 1;             // batch 数（每通道段数）
    int64_t c_ = 1;             // 通道数
    int64_t dhw_ = 1;           // D * H * W
    int64_t blk_ = BN3D_UB_ELEMS;
    int64_t channelStart_ = 0;  // 本核负责的通道区间 [channelStart_, channelEnd_)
    int64_t channelEnd_ = 0;
    int64_t localChunk_ = 0;    // 本核实际负责的通道数
    float epsilon_ = 1e-5f;
    float invCount_ = 1.0f;     // 1 / (N * DHW)
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void BatchNorm3D<T>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean, GM_ADDR save_variance, const BatchNorm3DTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 1. 解包 TilingData
    uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);

    dhw_ = static_cast<int64_t>(blockFactor & DHW_MASK);
    c_ = static_cast<int64_t>((blockFactor >> 39) & C_MASK);
    n_ = static_cast<int64_t>((ubFactor >> 32) & N_MASK);

    uint32_t epsBits = static_cast<uint32_t>(ubFactor & 0xFFFFFFFFULL);
    union { uint32_t u; float f; } epsConv;
    epsConv.u = epsBits;
    epsilon_ = epsConv.f;

    // 占位/退化 tiling 的一致性保护，避免除零、死循环与非法 shape
    if (dhw_ <= 0) { dhw_ = 1; }
    if (c_ <= 0) { c_ = 1; }
    if (n_ <= 0) { n_ = 1; }
    if (!(epsilon_ > 0.0f)) { epsilon_ = 1e-5f; }  // 处理 0 / 负数 / NaN

    invCount_ = 1.0f / static_cast<float>(n_ * dhw_);
    blk_ = BN3D_UB_ELEMS;

    int64_t totalNum = n_ * c_ * dhw_;
    if (totalNum <= 0) { totalNum = 1; }

    // 2. GlobalTensor 绑定
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);
    scaleGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), c_);
    offsetGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset), c_);
    saveMeanGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_mean), c_);
    saveVarGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_variance), c_);

    // 3. 多核切分：按通道均分
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockNum <= 0) { blockNum = 1; }
    int64_t chunk = (c_ + blockNum - 1) / blockNum;
    channelStart_ = blockIdx * chunk;
    channelEnd_ = channelStart_ + chunk;
    if (channelStart_ > c_) { channelStart_ = c_; }
    if (channelEnd_ > c_) { channelEnd_ = c_; }
    localChunk_ = channelEnd_ - channelStart_;

    // 4. UB 初始化
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(T)));
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(floatBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    }
    pipe.InitBuffer(sqBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(accBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(accSqBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(redBuf_, 256);
    pipe.InitBuffer(tmpBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));

    // 本核统计量缓冲（32B 对齐，最小 32B）
    uint32_t statBytes = 32U;
    if (localChunk_ > 0) {
        statBytes = static_cast<uint32_t>(((localChunk_ + 7) / 8) * 8) * static_cast<uint32_t>(sizeof(float));
    }
    pipe.InitBuffer(statMeanBuf_, statBytes);
    pipe.InitBuffer(statVarBuf_, statBytes);
}

template <typename T>
__aicore__ inline void BatchNorm3D<T>::CopyIn(int64_t gmOffset, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[gmOffset], extParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void BatchNorm3D<T>::Accumulate(int64_t currentNum)
{
    // 累加统计量：acc += x，accSq += x * x（均为 fp32 累加）
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<float> sq = sqBuf_.Get<float>();
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> accSq = accSqBuf_.Get<float>();
    int32_t cnt = static_cast<int32_t>(currentNum);
    if constexpr (sizeof(T) == 4) {
        Add(acc, acc, xLocal, cnt);
        Mul(sq, xLocal, xLocal, cnt);
        Add(accSq, accSq, sq, cnt);
    } else {
        LocalTensor<float> f = floatBuf_.Get<float>();
        Cast(f, xLocal, RoundMode::CAST_NONE, cnt);
        Add(acc, acc, f, cnt);
        Mul(sq, f, f, cnt);
        Add(accSq, accSq, sq, cnt);
    }
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void BatchNorm3D<T>::Normalize(int64_t currentNum, float coefA, float coefB)
{
    // 归一化与仿射：y = x * A + B，其中 A = invStd * scale，B = offset - mean * A
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    int32_t cnt = static_cast<int32_t>(currentNum);
    if constexpr (sizeof(T) == 4) {
        Muls(yLocal, xLocal, coefA, cnt);
        Adds(yLocal, yLocal, coefB, cnt);
    } else {
        LocalTensor<float> f = floatBuf_.Get<float>();
        Cast(f, xLocal, RoundMode::CAST_NONE, cnt);
        Muls(f, f, coefA, cnt);
        Adds(f, f, coefB, cnt);
        Cast(yLocal, f, RoundMode::CAST_RINT, cnt);
    }
    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void BatchNorm3D<T>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[gmOffset], yLocal, extParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline float BatchNorm3D<T>::ReduceBuffer(const LocalTensor<float>& buf)
{
    // 水平归约 buf[0, blk_) 求和，返回标量（V -> S 同步后经 GetValue 取回）
    LocalTensor<float> red = redBuf_.Get<float>();
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    ReduceSum(red, buf, tmp, static_cast<int32_t>(blk_));
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    float result = red.GetValue(0);
    return result;
}

template <typename T>
__aicore__ inline void BatchNorm3D<T>::WriteChannelStats(int64_t localCount)
{
    // 将本核各通道的 mean / var 经 MTE3 写出（避免逐核 GlobalTensor::SetValue 写 GM 丢失）
    if (localCount <= 0) {
        return;
    }
    LocalTensor<float> meanLocal = statMeanBuf_.Get<float>();
    LocalTensor<float> varLocal = statVarBuf_.Get<float>();
    event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
    DataCopyExtParams extParams{1, static_cast<uint32_t>(localCount * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    DataCopyPad(saveMeanGM[channelStart_], meanLocal, extParams);
    DataCopyPad(saveVarGM[channelStart_], varLocal, extParams);
}

template <typename T>
__aicore__ inline void BatchNorm3D<T>::Process()
{
    // TODO: 实现 Process 逻辑
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> accSq = accSqBuf_.Get<float>();
    LocalTensor<float> meanLocal = statMeanBuf_.Get<float>();
    LocalTensor<float> varLocal = statVarBuf_.Get<float>();

    for (int64_t c = channelStart_; c < channelEnd_; ++c) {
        // ---------------- pass 1：统计 sum / sum of squares ----------------
        // 累加器跨段不清零：位置上 p 的元素映射到 acc[p % blk_]，
        // 因此所有段处理完后一次归约即可得到整通道的 sum / sumsq。
        Duplicate(acc, 0.0f, static_cast<int32_t>(blk_));
        Duplicate(accSq, 0.0f, static_cast<int32_t>(blk_));
        for (int64_t nb = 0; nb < n_; ++nb) {
            int64_t base = (nb * c_ + c) * dhw_;
            for (int64_t o = 0; o < dhw_; o += blk_) {
                int64_t cur = dhw_ - o;
                if (cur > blk_) {
                    cur = blk_;
                }
                CopyIn(base + o, cur);
                Accumulate(cur);
            }
        }
        float sum = ReduceBuffer(acc);
        float sumsq = ReduceBuffer(accSq);

        float mean = sum * invCount_;
        float var = sumsq * invCount_ - mean * mean;
        if (var < 0.0f) {
            var = 0.0f;
        }
        float invStd = 1.0f / sqrt(var + epsilon_);
        float sc = scaleGM.GetValue(static_cast<uint32_t>(c));
        float of = offsetGM.GetValue(static_cast<uint32_t>(c));
        float coefA = invStd * sc;
        float coefB = of - mean * coefA;

        // 本核统计量写入 UB，循环结束后统一经 MTE3 写出
        meanLocal.SetValue(static_cast<int32_t>(c - channelStart_), mean);
        varLocal.SetValue(static_cast<int32_t>(c - channelStart_), var);

        // ---------------- pass 2：归一化 + 仿射 ----------------
        for (int64_t nb = 0; nb < n_; ++nb) {
            int64_t base = (nb * c_ + c) * dhw_;
            for (int64_t o = 0; o < dhw_; o += blk_) {
                int64_t cur = dhw_ - o;
                if (cur > blk_) {
                    cur = blk_;
                }
                CopyIn(base + o, cur);
                Normalize(cur, coefA, coefB);
                CopyOut(base + o, cur);
            }
        }
    }

    // 本核通道统计量经 MTE3 写出
    WriteChannelStats(localChunk_);

    // 确保所有 MTE3（GM 写出）在 kernel 退出前完成
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
}

} // namespace NsBatchNorm3D
#endif // BATCHNORM3D_H