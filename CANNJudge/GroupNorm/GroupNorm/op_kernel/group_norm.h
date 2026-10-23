/*!
 * \file group_norm.h
 * \brief GroupNorm 算子 kernel 类定义
 */

#ifndef GROUPNORM_H
#define GROUPNORM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_tiling_data.h"
#include "group_norm_tiling_key.h"

namespace NsGroupNorm {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 单块 UB 处理元素数（编译期固定，保证 UB 不溢出）
constexpr int64_t GN_UB_ELEMS = 4096;

// TilingData 位域约定（结构不可改，host / kernel 两侧严格一致）：
//   blockFactor : bit[0:32)  = hwSize（x 中 C 之后所有维度的乘积）
//                 bit[32:48) = C（通道数）
//                 bit[48:64) = G（num_groups，分组数）
//   ubFactor    : bit[0:32)  = epsilon 的 IEEE-754 位模式
//                 bit[32:64) = N（batch 数）
//   totalNum    : 保留 = N * C * hwSize（kernel 由位域字段推导，不依赖该字段）
constexpr uint64_t HWSIZE_MASK = 0xFFFFFFFFULL;  // 32 bit
constexpr uint64_t C_MASK = 0xFFFFULL;           // 16 bit
constexpr uint64_t G_MASK = 0xFFFFULL;           // 16 bit
constexpr uint64_t N_MASK = 0xFFFFFFFFULL;       // 32 bit

template <typename T>
class GroupNorm {
public:
    __aicore__ inline GroupNorm(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, const GroupNormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void Accumulate(int64_t currentNum);
    __aicore__ inline void Normalize(int64_t currentNum, float coefA, float coefB);
    __aicore__ inline float ReduceBuffer(const LocalTensor<float>& buf);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> floatBuf_;   // fp16 输入升 fp32 的中间缓冲
    TBuf<TPosition::VECCALC> sqBuf_;      // x * x 的中间缓冲
    TBuf<TPosition::VECCALC> accBuf_;     // sum 累加器
    TBuf<TPosition::VECCALC> accSqBuf_;   // sum of squares 累加器
    TBuf<TPosition::VECCALC> redBuf_;     // 归约结果
    TBuf<TPosition::VECCALC> tmpBuf_;     // 归约 sharedTmpBuffer

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> gammaGM;
    GlobalTensor<float> betaGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;

    int64_t n_ = 1;            // batch 数
    int64_t c_ = 1;            // 通道数
    int64_t g_ = 1;            // 分组数（num_groups）
    int64_t cg_ = 1;           // 每组通道数 = C / G
    int64_t hwSize_ = 1;       // C 之后所有维度的乘积
    int64_t groupSize_ = 1;    // 每组归约元素数 = cg_ * hwSize_
    int64_t blk_ = GN_UB_ELEMS;
    int64_t groupStart_ = 0;   // 本核负责的组区间 [groupStart_, groupEnd_)
    int64_t groupEnd_ = 0;
    float epsilon_ = 1e-5f;
    float invGroupSize_ = 1.0f;  // 1 / groupSize_
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void GroupNorm<T>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, const GroupNormTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 1. 解包 TilingData
    uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);

    hwSize_ = static_cast<int64_t>(blockFactor & HWSIZE_MASK);
    c_ = static_cast<int64_t>((blockFactor >> 32) & C_MASK);
    g_ = static_cast<int64_t>((blockFactor >> 48) & G_MASK);
    n_ = static_cast<int64_t>((ubFactor >> 32) & N_MASK);

    uint32_t epsBits = static_cast<uint32_t>(ubFactor & 0xFFFFFFFFULL);
    union { uint32_t u; float f; } epsConv;
    epsConv.u = epsBits;
    epsilon_ = epsConv.f;

    // 占位/退化 tiling 的一致性保护，避免除零、死循环与非法 shape
    if (hwSize_ <= 0) { hwSize_ = 1; }
    if (c_ <= 0) { c_ = 1; }
    if (g_ <= 0) { g_ = 1; }
    if (n_ <= 0) { n_ = 1; }
    if (!(epsilon_ > 0.0f)) { epsilon_ = 1e-5f; }  // 处理 0 / 负数 / NaN

    cg_ = c_ / g_;
    if (cg_ <= 0) { cg_ = 1; }
    groupSize_ = cg_ * hwSize_;
    if (groupSize_ <= 0) { groupSize_ = 1; }
    invGroupSize_ = 1.0f / static_cast<float>(groupSize_);
    blk_ = GN_UB_ELEMS;

    int64_t totalNum = n_ * c_ * hwSize_;
    if (totalNum <= 0) { totalNum = 1; }

    // 2. GlobalTensor 绑定
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);
    gammaGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma), c_);
    betaGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta), c_);

    // 3. 多核切分：按组（N * G 个组）均分，组内归约天然无跨核依赖
    int64_t totalGroups = n_ * g_;
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockNum <= 0) { blockNum = 1; }
    int64_t chunk = (totalGroups + blockNum - 1) / blockNum;
    groupStart_ = blockIdx * chunk;
    groupEnd_ = groupStart_ + chunk;
    if (groupStart_ > totalGroups) { groupStart_ = totalGroups; }
    if (groupEnd_ > totalGroups) { groupEnd_ = totalGroups; }
    blockLength_ = groupEnd_ - groupStart_;
    ubLength_ = blk_;

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
}

template <typename T>
__aicore__ inline void GroupNorm<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress], extParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::Accumulate(int64_t currentNum)
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
__aicore__ inline void GroupNorm<T>::Normalize(int64_t currentNum, float coefA, float coefB)
{
    // 归一化与仿射：y = x * A + B，其中 A = invStd * gamma，B = beta - mean * A
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
__aicore__ inline void GroupNorm<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    // 归一化主体在 Normalize() 中实现，该占位函数保持接口完整
    (void)currentNum;
}

template <typename T>
__aicore__ inline void GroupNorm<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[progress], yLocal, extParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline float GroupNorm<T>::ReduceBuffer(const LocalTensor<float>& buf)
{
    // 水平归约 buf[0, blk_) 求和，返回标量
    // reduce 类指令写回后先 PipeBarrier<PIPE_V> drain，再经 V_S 同步取回标量
    LocalTensor<float> red = redBuf_.Get<float>();
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    ReduceSum(red, buf, tmp, static_cast<int32_t>(blk_));
    PipeBarrier<PIPE_V>();
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    float result = red.GetValue(0);
    return result;
}

template <typename T>
__aicore__ inline void GroupNorm<T>::Process()
{
    // TODO: 实现 Process 逻辑
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> accSq = accSqBuf_.Get<float>();

    for (int64_t gIdx = groupStart_; gIdx < groupEnd_; ++gIdx) {
        // 组 gIdx = n * G + g，其元素在 x 中为连续区间 [gIdx*groupSize_, +groupSize_)
        int64_t groupBase = gIdx * groupSize_;

        // ---------------- pass 1：统计 sum / sum of squares ----------------
        // 累加器跨块不清零：位置上 p 的元素映射到 acc[p % blk_]，
        // 因此所有块处理完后一次归约即可得到整组的 sum / sumsq。
        Duplicate(acc, 0.0f, static_cast<int32_t>(blk_));
        Duplicate(accSq, 0.0f, static_cast<int32_t>(blk_));
        for (int64_t o = 0; o < groupSize_; o += blk_) {
            int64_t cur = groupSize_ - o;
            if (cur > blk_) {
                cur = blk_;
            }
            CopyIn(groupBase + o, cur);
            Accumulate(cur);
        }
        float sum = ReduceBuffer(acc);
        float sumsq = ReduceBuffer(accSq);

        float mean = sum * invGroupSize_;
        float var = sumsq * invGroupSize_ - mean * mean;
        if (var < 0.0f) {
            var = 0.0f;
        }
        float invStd = 1.0f / sqrt(var + epsilon_);

        // ---------------- pass 2：按通道归一化 + 仿射 ----------------
        // gamma / beta 是通道级的，同一组内各通道系数不同，故按通道循环
        int64_t n = gIdx / g_;
        int64_t g = gIdx - n * g_;
        for (int64_t ci = 0; ci < cg_; ++ci) {
            int64_t c = g * cg_ + ci;
            float gm = gammaGM.GetValue(static_cast<uint32_t>(c));
            float bt = betaGM.GetValue(static_cast<uint32_t>(c));
            float coefA = invStd * gm;
            float coefB = bt - mean * coefA;

            int64_t channelBase = (n * c_ + c) * hwSize_;
            for (int64_t o = 0; o < hwSize_; o += blk_) {
                int64_t cur = hwSize_ - o;
                if (cur > blk_) {
                    cur = blk_;
                }
                CopyIn(channelBase + o, cur);
                Normalize(cur, coefA, coefB);
                CopyOut(channelBase + o, cur);
            }
        }
    }

    // 确保所有 MTE3（GM 写出）在 kernel 退出前完成
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
}

} // namespace NsGroupNorm
#endif // GROUPNORM_H
