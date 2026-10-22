/*!
 * \file bn_training_update.h
 * \brief BnTrainingUpdate 算子 kernel 类定义
 */

#ifndef BNTRAININGUPDATE_H
#define BNTRAININGUPDATE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bn_training_update_tiling_data.h"
#include "bn_training_update_tiling_key.h"

namespace NsBnTrainingUpdate {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// TilingData 字段的位域约定（结构不变，语义由 host/kernel 两侧约定）：
//   blockFactor : bit[0:21)  = hw = H * W（同一 batch、同一通道内的元素数）
//                 bit[21:34) = C（通道数）
//                 bit[34:59) = rows = N * C
//   ubFactor    : bit[0:32)  = epsilon 的 IEEE-754 位模式
//   totalNum    : 保留（元素总数 N*C*H*W，kernel 由 rows*hw 推导，不依赖该字段）
constexpr uint64_t HW_MASK = 0x1FFFFFULL;
constexpr uint64_t C_MASK = 0x1FFFULL;
constexpr uint64_t ROWS_MASK = 0x1FFFFFFULL;
// 单轮 UB 处理元素数（编译期固定，避免依赖 tiling 字段且保证 UB 不溢出）
constexpr int64_t UB_ELEMS_FIXED = 4096;

template <typename T>
class BnTrainingUpdate {
public:
    __aicore__ inline BnTrainingUpdate(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean, GM_ADDR save_variance, const BnTrainingUpdateTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void WriteChannelStats();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> floatBuf_;  // fp16 输入时的 fp32 计算中间缓冲

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> sumGM;
    GlobalTensor<float> squareSumGM;
    GlobalTensor<float> scaleGM;
    GlobalTensor<float> offsetGM;
    GlobalTensor<float> saveMeanGM;
    GlobalTensor<float> saveVarGM;

    int64_t channelNum_ = 1;       // C
    int64_t hw_ = 1;               // H * W
    int64_t ubLength_ = 1;         // 单轮 UB 元素数
    int64_t totalNum_ = 1;         // N * C * H * W
    int64_t perChannelElem_ = 1;   // N * H * W
    int64_t blockNum_ = 1;
    int64_t blockIdx_ = 0;
    int64_t gStart_ = 0;
    int64_t gEnd_ = 0;

    float invPerChannelElem_ = 1.0f;
    float epsilon_ = 1e-5f;

    float curA_ = 0.0f;
    float curB_ = 0.0f;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::Init(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR save_mean, GM_ADDR save_variance, const BnTrainingUpdateTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 1. 解包 TilingData
    uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    hw_ = static_cast<int64_t>(blockFactor & HW_MASK);
    channelNum_ = static_cast<int64_t>((blockFactor >> 21) & C_MASK);
    int64_t rows = static_cast<int64_t>((blockFactor >> 34) & ROWS_MASK);

    uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);
    uint32_t epsBits = static_cast<uint32_t>(ubFactor & 0xFFFFFFFFULL);
    ubLength_ = UB_ELEMS_FIXED;

    union { uint32_t u; float f; } epsConv;
    epsConv.u = epsBits;
    epsilon_ = epsConv.f;

    // 占位/退化 tiling 的一致性保护，避免除零与死循环
    if (hw_ <= 0) { hw_ = 1; }
    if (channelNum_ <= 0) { channelNum_ = 1; }
    if (rows <= 0) { rows = 1; }
    totalNum_ = rows * hw_;
    if (totalNum_ <= 0) { totalNum_ = 1; }

    perChannelElem_ = totalNum_ / channelNum_;   // = N * H * W
    if (perChannelElem_ <= 0) { perChannelElem_ = 1; }
    invPerChannelElem_ = 1.0f / static_cast<float>(perChannelElem_);

    // 2. 多核切分：按元素总量均分连续区间 [gStart_, gEnd_)
    blockNum_ = static_cast<int64_t>(GetBlockNum());
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    if (blockNum_ <= 0) { blockNum_ = 1; }
    int64_t chunk = (totalNum_ + blockNum_ - 1) / blockNum_;
    gStart_ = blockIdx_ * chunk;
    gEnd_ = gStart_ + chunk;
    if (gStart_ > totalNum_) { gStart_ = totalNum_; }
    if (gEnd_ > totalNum_) { gEnd_ = totalNum_; }

    // 3. GlobalTensor 绑定
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum_);
    sumGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum), channelNum_);
    squareSumGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(square_sum), channelNum_);
    scaleGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), channelNum_);
    offsetGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset), channelNum_);
    saveMeanGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_mean), channelNum_);
    saveVarGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(save_variance), channelNum_);

    // 4. UB 初始化
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(floatBuf_, ubLength_ * static_cast<int64_t>(sizeof(float)));
    }

    // 5. save_mean / save_variance 由 block0 独占写出（值确定性一致）
    if (blockIdx_ == 0) {
        WriteChannelStats();
    }
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::WriteChannelStats()
{
    // 直接由 sum / square_sum 标量换算 mean / var（与 golden 一致的有偏方差）
    for (int64_t c = 0; c < channelNum_; ++c) {
        float s = sumGM.GetValue(static_cast<uint32_t>(c));
        float sq = squareSumGM.GetValue(static_cast<uint32_t>(c));
        float mean = s * invPerChannelElem_;
        float var = sq * invPerChannelElem_ - mean * mean;
        saveMeanGM.SetValue(static_cast<uint32_t>(c), mean);
        saveVarGM.SetValue(static_cast<uint32_t>(c), var);
    }
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress], extParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    if constexpr (sizeof(T) == 4) {
        // fp32：直接向量化 y = x * a + b
        Muls(yLocal, xLocal, curA_, currentNum);
        Adds(yLocal, yLocal, curB_, currentNum);
    } else {
        // fp16：升 fp32 计算后降回 fp16
        LocalTensor<float> f = floatBuf_.Get<float>();
        Cast(f, xLocal, RoundMode::CAST_NONE, currentNum);
        Muls(f, f, curA_, currentNum);
        Adds(f, f, curB_, currentNum);
        Cast(yLocal, f, RoundMode::CAST_RINT, currentNum);
    }
    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[progress], yLocal, extParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void BnTrainingUpdate<T>::Process()
{
    // TODO: 实现 Process 逻辑
    // 遍历本核负责的连续元素区间，按「行」切分（同一行共享同一通道参数）
    int64_t g = gStart_;
    while (g < gEnd_) {
        int64_t row = g / hw_;             // 行号 = batch * C + channel
        int64_t col = g % hw_;             // 行内偏移
        int64_t remain = hw_ - col;
        int64_t left = gEnd_ - g;
        if (remain > left) {
            remain = left;
        }

        // 由 GM 标量求该通道的归一化+仿射系数：y = x * a + b
        int64_t channel = row % channelNum_;
        float s = sumGM.GetValue(static_cast<uint32_t>(channel));
        float sq = squareSumGM.GetValue(static_cast<uint32_t>(channel));
        float sc = scaleGM.GetValue(static_cast<uint32_t>(channel));
        float of = offsetGM.GetValue(static_cast<uint32_t>(channel));
        float mean = s * invPerChannelElem_;
        float var = sq * invPerChannelElem_ - mean * mean;
        float invStd = 1.0f / sqrt(var + epsilon_);
        curA_ = invStd * sc;
        curB_ = of - mean * curA_;

        int64_t done = 0;
        while (done < remain) {
            int64_t cur = remain - done;
            if (cur > ubLength_) {
                cur = ubLength_;
            }
            CopyIn(g + done, cur);
            Compute(cur);
            CopyOut(g + done, cur);
            done += cur;
        }
        g += remain;
    }
    // 确保所有 MTE3（GM 写出）在 kernel 退出前完成
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
}

} // namespace NsBnTrainingUpdate
#endif // BNTRAININGUPDATE_H