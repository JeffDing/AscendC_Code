/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// negative_slope 属性缺失时的 PyTorch 默认值
constexpr float LEAKY_RELU_DEFAULT_SLOPE = 0.01f;
// ubFactor 高 32 位为 negative_slope 的 IEEE-754 位模式；位模式 0 表示"未设置"，回退默认值
constexpr uint64_t LEAKY_RELU_SLOPE_ZERO_CODE = 0U;
// host 对 negative_slope == 0 编码为最小次正规位（数值 0），把位模式 0 留作"未设置"哨兵
constexpr uint32_t LEAKY_RELU_SLOPE_DENORM_ONE = 1U;
// 每轮 UB 元素数上限：双缓冲输入输出 + (half)float32 中间缓冲共 16 字节/元素，
// 8192 元素对应 128KB，低于 192KB UB 上限，防止退化 tiling 超限
constexpr int64_t UB_ELEMS_MAX = 8192;

template <typename T>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // half 输入时用于 float32 中间计算的缓冲（float 输入直接原地计算，不分配）
    TBuf<TPosition::VECCALC> calcBuf_;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t blockIdx_ = 0;
    int64_t offset_ = 0;
    int64_t myCount_ = 0;
    float slope_ = LEAKY_RELU_DEFAULT_SLOPE;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 1) 解包 TilingData。字段为标量直存，避免位域拆分引入的退化 tiling 歧义
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = static_cast<uint32_t>(static_cast<uint64_t>(tilingData->ubFactor) & 0xFFFFFFFFULL);
    uint32_t slopeCode = static_cast<uint32_t>(static_cast<uint64_t>(tilingData->ubFactor) >> 32);

    // 占位/退化 tiling 保护：字段为 0 或越界时回退安全值，保证循环与内存安全
    if (totalNum_ < 1) {
        totalNum_ = 1;
    }
    if (blockLength_ < 1) {
        blockLength_ = totalNum_;
    }
    if (ubLength_ < 1) {
        ubLength_ = 1;
    }
    if (ubLength_ > UB_ELEMS_MAX) {
        // 占位 tiling 可能给出远超 UB 容量的分块值，钳制到安全上限再申请缓冲
        ubLength_ = UB_ELEMS_MAX;
    }
    if (ubLength_ > totalNum_) {
        ubLength_ = totalNum_;
    }
    // slope 位模式解码：0=未设置(回退默认 0.01)，1=host 编码的 slope==0(数值 0)，其余直接取位模式
    if (slopeCode == LEAKY_RELU_SLOPE_ZERO_CODE) {
        slope_ = LEAKY_RELU_DEFAULT_SLOPE;
    } else if (slopeCode == LEAKY_RELU_SLOPE_DENORM_ONE) {
        slope_ = 0.0f;
    } else {
        union { uint32_t u; float f; } slopeConv;
        slopeConv.u = slopeCode;
        slope_ = slopeConv.f;
    }

    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    offset_ = blockIdx_ * blockLength_;
    myCount_ = totalNum_ - offset_;
    if (myCount_ > blockLength_) {
        myCount_ = blockLength_;
    }
    if (myCount_ < 0) {
        myCount_ = 0;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint32_t>(totalNum_));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), static_cast<uint32_t>(totalNum_));

    // 2) 申请 UB：输入/输出双缓冲 + (half) float32 中间缓冲，与 host 侧 16 字节/元素预算一致
    const uint32_t elemBytes = static_cast<uint32_t>(sizeof(T));
    const uint32_t slotBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(elemBytes));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, slotBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, slotBytes);
    if constexpr (sizeof(T) == 2) {
        // half：经 float32 中间路径，与 golden 的 float64 中间计算对齐（fp16 单次舍入）
        pipe.InitBuffer(calcBuf_, static_cast<uint32_t>(ubLength_ * 4));
    }
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> x = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(x, inputGMX[offset_ + progress], copyParams, padParams);
    inputQueueX.EnQue<T>(x);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    // count 直接用 currentNum：LeakyRelu/Cast Level 2 接口内部按 mask 处理非对齐尾部
    const int32_t cnt = static_cast<int32_t>(currentNum);
    LocalTensor<T> x = inputQueueX.DeQue<T>();
    LocalTensor<T> y = outputQueueY.AllocTensor<T>();

    if constexpr (sizeof(T) == 2) {
        // half：x -> float32 -> LeakyRelu(float32) -> float32 -> half
        // 全程 float32 中间精度，输出仅一次 half 舍入，与参考实现 (float64 -> fp16) 逐位一致
        auto calc = calcBuf_.Get<float>();
        Cast(calc, x, RoundMode::CAST_NONE, cnt);
        LeakyRelu(calc, calc, slope_, cnt);
        Cast(y, calc, RoundMode::CAST_NONE, cnt);
    } else {
        // float：直接原地计算（golden 为 float64 中间再舍回 float32，误差 <= 1 ulp，
        // 远小于 1e-4 的 rtol/atol 判定阈值）
        LeakyRelu(y, x, slope_, cnt);
    }

    inputQueueX.FreeTensor(x);
    outputQueueY.EnQue<T>(y);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> y = outputQueueY.DeQue<T>();
    DataCopyExtParams outParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[offset_ + progress], y, outParams);
    outputQueueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    for (int64_t r = 0; r < myCount_; r += ubLength_) {
        int64_t cur = myCount_ - r;
        if (cur > ubLength_) {
            cur = ubLength_;
        }
        CopyIn(r, cur);
        Compute(cur);
        CopyOut(r, cur);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
