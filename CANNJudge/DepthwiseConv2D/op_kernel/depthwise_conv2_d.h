/*!
 * \file depthwise_conv2_d.h
 * \brief DepthwiseConv2D 算子 kernel 类定义
 */

#ifndef DEPTHWISECONV2D_H
#define DEPTHWISECONV2D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "depthwise_conv2_d_tiling_data.h"
#include "depthwise_conv2_d_tiling_key.h"

#include <type_traits>

namespace NsDepthwiseConv2D {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 输出行单块最大列数（编译期固定，保证 UB 不溢出）。
constexpr int64_t OW_BLK_MAX = 4096;

// 向量指令 count 参数的 32B 对齐。
// 真机实证（多个已过评测算子一致）：count 模式按 32B 块粒度处理，尾部不足一个 32B 块的
// 元素不生效，因此 count 必须向上取整到 32B 边界；UB 各子区起始地址均已 32B 对齐，
// 向上取整后越界的 pad 元素由 DataCopyPad 的 rightPadding 补 0，不参与最终输出。
template <typename T>
__aicore__ inline int32_t VecAlignCount(int64_t n)
{
    constexpr int64_t kElemsPer32B = 32 / static_cast<int64_t>(sizeof(T));
    return static_cast<int32_t>((n + kElemsPer32B - 1) & ~(kElemsPer32B - 1));
}

// TilingData 位域约定（与 op_host/depthwise_conv2_d_tiling.cpp 严格一致）。
// 刻意不把任何关键标量放在 totalNum：真实框架下 TilingData 首字段在 kernel 侧读取不可靠
// （偶发读到 0 / 8，已由常量 tiling 受控实验确证），故 kernel 完全不读 totalNum。
//   blockFactor : bit[0:13) = N, bit[13:26) = C, bit[26:39) = H_in, bit[39:50) = W_in
//   ubFactor    : bit[0:4) = kH, bit[4:8) = kW, bit[8:20) = stride,
//                 bit[20:32) = pad, bit[32:44) = dilation, bit[44:45) = hasBias
constexpr uint64_t BF_N_SHIFT = 0;
constexpr uint64_t BF_C_SHIFT = 13;
constexpr uint64_t BF_HIN_SHIFT = 26;
constexpr uint64_t BF_WIN_SHIFT = 39;
constexpr uint64_t BF_N_MASK = 0x1FFFULL;     // 13 bit
constexpr uint64_t BF_C_MASK = 0x1FFFULL;    // 13 bit
constexpr uint64_t BF_HIN_MASK = 0x1FFFULL;  // 13 bit
constexpr uint64_t BF_WIN_MASK = 0x7FFULL;   // 11 bit

constexpr uint64_t UF_KH_SHIFT = 0;
constexpr uint64_t UF_KW_SHIFT = 4;
constexpr uint64_t UF_STRIDE_SHIFT = 8;
constexpr uint64_t UF_PAD_SHIFT = 20;
constexpr uint64_t UF_DIL_SHIFT = 32;
constexpr uint64_t UF_HASBIAS_SHIFT = 44;
constexpr uint64_t UF_K_MASK = 0xFULL;          // 4 bit
constexpr uint64_t UF_PARAM_MASK = 0xFFFULL;    // 12 bit

template <typename T>
class DepthwiseConv2D {
public:
    __aicore__ inline DepthwiseConv2D(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, const DepthwiseConv2DTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

    // 单个 (n, c) 输出平面的完整计算。
    __aicore__ inline void ProcessChannel(int64_t n, int64_t c);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // 计算用 UB。outFBuf_ 以 offset 0 参与向量指令，规避真机上
    // "dst 带运行期 offset 的向量指令只有 offset=0 生效"的缺陷。
    TBuf<TPosition::VECCALC> xFBuf_;     // x 行升 float 后的缓冲 [wInAlF]
    TBuf<TPosition::VECCALC> xSegBuf_;   // x 段 float 缓冲 [owBlkAl]
    TBuf<TPosition::VECCALC> outFBuf_;   // 输出行 float 缓冲 [owBlkAlF]
    TBuf<TPosition::VECCALC> wBuf_;      // w[c,0,:,:] 缓冲（T 类型）[kH*kW 对齐]

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> weightGMW;
    GlobalTensor<float> biasGMB;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;

    int64_t n_ = 1;
    int64_t c_ = 1;
    int64_t hIn_ = 1;
    int64_t wIn_ = 1;
    int64_t kH_ = 1;
    int64_t kW_ = 1;
    int64_t stride_ = 1;
    int64_t pad_ = 0;
    int64_t dil_ = 1;
    int64_t hasBias_ = 0;

    int64_t hOut_ = 1;
    int64_t wOut_ = 1;
    int64_t planeIn_ = 1;    // H_in * W_in
    int64_t planeOut_ = 1;   // H_out * W_out
    int64_t totalTask_ = 1;  // N * C

    int64_t taskStart_ = 0;
    int64_t taskEnd_ = 0;

    int64_t owBlkMax_ = 1;  // 实际分配的输出行块上限
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, const DepthwiseConv2DTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 1. 解包 TilingData（位域与 op_host tiling.cpp 一致；不读 totalNum）
    uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);

    n_ = static_cast<int64_t>((blockFactor >> BF_N_SHIFT) & BF_N_MASK);
    c_ = static_cast<int64_t>((blockFactor >> BF_C_SHIFT) & BF_C_MASK);
    hIn_ = static_cast<int64_t>((blockFactor >> BF_HIN_SHIFT) & BF_HIN_MASK);
    wIn_ = static_cast<int64_t>((blockFactor >> BF_WIN_SHIFT) & BF_WIN_MASK);

    kH_ = static_cast<int64_t>((ubFactor >> UF_KH_SHIFT) & UF_K_MASK);
    kW_ = static_cast<int64_t>((ubFactor >> UF_KW_SHIFT) & UF_K_MASK);
    stride_ = static_cast<int64_t>((ubFactor >> UF_STRIDE_SHIFT) & UF_PARAM_MASK);
    pad_ = static_cast<int64_t>((ubFactor >> UF_PAD_SHIFT) & UF_PARAM_MASK);
    dil_ = static_cast<int64_t>((ubFactor >> UF_DIL_SHIFT) & UF_PARAM_MASK);
    hasBias_ = static_cast<int64_t>((ubFactor >> UF_HASBIAS_SHIFT) & 0x1ULL);

    // 占位 / 退化 tiling 的一致性保护（本地 UT 用占位值，须保证不除零、不死循环、不越界）
    if (n_ < 1) { n_ = 1; }
    if (c_ < 1) { c_ = 1; }
    if (hIn_ < 1) { hIn_ = 1; }
    if (wIn_ < 1) { wIn_ = 1; }
    if (kH_ < 1) { kH_ = 1; }
    if (kW_ < 1) { kW_ = 1; }
    if (stride_ < 1) { stride_ = 1; }
    if (dil_ < 1) { dil_ = 1; }
    if (pad_ < 0) { pad_ = 0; }

    // H_out = (H_in + 2*pad - dil*(kH-1) - 1) / stride + 1
    hOut_ = (hIn_ + 2 * pad_ - dil_ * (kH_ - 1) - 1) / stride_ + 1;
    wOut_ = (wIn_ + 2 * pad_ - dil_ * (kW_ - 1) - 1) / stride_ + 1;
    if (hOut_ < 1) { hOut_ = 1; }
    if (wOut_ < 1) { wOut_ = 1; }

    planeIn_ = hIn_ * wIn_;
    planeOut_ = hOut_ * wOut_;
    totalTask_ = n_ * c_;

    // 2. GlobalTensor 绑定
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint32_t>(n_ * c_ * planeIn_));
    weightGMW.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(w), static_cast<uint32_t>(c_ * kH_ * kW_));
    biasGMB.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint32_t>(c_));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), static_cast<uint32_t>(n_ * c_ * planeOut_));

    // 3. 多核切分：任务 = (n, c) 输出平面，任务间无数据依赖，按任务均分、无跨核竞争
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockNum < 1) { blockNum = 1; }
    int64_t chunk = (totalTask_ + blockNum - 1) / blockNum;
    if (chunk < 1) { chunk = 1; }
    taskStart_ = blockIdx * chunk;
    taskEnd_ = taskStart_ + chunk;
    if (taskStart_ > totalTask_) { taskStart_ = totalTask_; }
    if (taskEnd_ > totalTask_) { taskEnd_ = totalTask_; }
    blockLength_ = taskEnd_ - taskStart_;

    // 4. UB 分配
    owBlkMax_ = wOut_;
    if (owBlkMax_ > OW_BLK_MAX) { owBlkMax_ = OW_BLK_MAX; }
    if (owBlkMax_ < 1) { owBlkMax_ = 1; }

    // 向量指令（Cast/Muls/Add/Duplicate）在 CPU 调试构建下要求 dst/src 张量至少 64 字节
    // （CheckFunVecBinaryScalarDiffType），小 shape（wOut 很小）时须把各子区抬到该下限，
    // 否则会 abort。代价至多每块 64B，对峰值占用无影响。
    constexpr int64_t kMinVecElems = 64 / static_cast<int64_t>(sizeof(T));
    constexpr int64_t kMinVecElemsF = 64 / static_cast<int64_t>(sizeof(float));
    int64_t xInAl = VecAlignCount<T>(wIn_);
    if (xInAl < kMinVecElems) { xInAl = kMinVecElems; }
    // xF 多分配 kMinVecElemsF 个元素，保证 stride==1 时 Muls 的 src offset + cntAl 不越界
    // （有效 cnt <= W_in，cntAl 向上取整可能多出最多 7 个元素，加上 16 的下限绰绰有余）
    int64_t xInAlF = VecAlignCount<float>(wIn_) + kMinVecElemsF;
    int64_t xSegAl = VecAlignCount<float>(owBlkMax_);
    if (xSegAl < kMinVecElemsF) { xSegAl = kMinVecElemsF; }
    int64_t owBlkAl = VecAlignCount<T>(owBlkMax_);
    if (owBlkAl < kMinVecElems) { owBlkAl = kMinVecElems; }
    int64_t owBlkAlF = VecAlignCount<float>(owBlkMax_);
    if (owBlkAlF < kMinVecElemsF) { owBlkAlF = kMinVecElemsF; }
    int64_t wAl = VecAlignCount<T>(kH_ * kW_);
    if (wAl < kMinVecElems) { wAl = kMinVecElems; }

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(xInAl * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(owBlkAl * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(xFBuf_, static_cast<uint32_t>(xInAlF * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(xSegBuf_, static_cast<uint32_t>(xSegAl * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(outFBuf_, static_cast<uint32_t>(owBlkAlF * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(wBuf_, static_cast<uint32_t>(wAl * static_cast<int64_t>(sizeof(T))));
    ubLength_ = xInAl;
}

// CastOrCopy<T>：T != float 时把 inputQueueX 中的 T 升为 float；T == float 时是同一类型。
// dav_c220 不支持 float→float 的 Cast（CAST_NONE 会直接 abort），故同型走 UB→UB DataCopy。
// 注意 blockLen 单位为 32B 块（DataCopyParams 语义），故按 sizeof(T) 归一。
template <typename T>
__aicore__ inline void CastOrCopy(LocalTensor<float>& dst, LocalTensor<T>& src, int64_t cnt)
{
    if constexpr (std::is_same<T, float>::value) {
        constexpr uint16_t kBlockLen = 32U / static_cast<uint16_t>(sizeof(float));
        DataCopy(dst, src, DataCopyParams{1U, static_cast<uint16_t>((cnt + kBlockLen - 1) / kBlockLen), 0U, 0U});
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, static_cast<uint32_t>(cnt));
    }
}

// CopyIn：把 x 的一段连续输入行加载到 inputQueueX。
// 尾部不足 32B 的部分由 DataCopyPad 的 rightPadding 补 0，使后续 Cast/Muls 的 count
// 可以安全取 32B 对齐值（pad 元素对累加结果贡献 0，不影响正确性）。
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    if (currentNum <= 0) {
        return;
    }
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    int32_t cntAl = VecAlignCount<T>(currentNum);
    DataCopyExtParams extParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))),
        0,
        0,
        0};
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(cntAl - currentNum), static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress], extParams, padParams);
    inputQueueX.EnQue(xLocal);
}

// Compute：把 outFBuf_ 中已累加的 float 结果转换为输出 dtype，送入 outputQueueY。
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    if (currentNum <= 0) {
        return;
    }
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> outF = outFBuf_.Get<float>();
    if constexpr (std::is_same<T, float>::value) {
        constexpr uint16_t kBlockLen = 32U / static_cast<uint16_t>(sizeof(float));
        DataCopy(yLocal, outF, DataCopyParams{1U, static_cast<uint16_t>((currentNum + kBlockLen - 1) / kBlockLen), 0U, 0U});
    } else {
        Cast(yLocal, outF, RoundMode::CAST_RINT, VecAlignCount<T>(currentNum));
    }
    outputQueueY.EnQue(yLocal);
}

// CopyOut：把 outputQueueY 中的整段输出行写出到 y（MTE3）。
// 全程连续写 GM，不用 GlobalTensor::SetValue——多核各自标量写 GM 会大面积丢失（已确证教训）。
template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    if (currentNum <= 0) {
        return;
    }
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams extParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))),
        0,
        0,
        0};
    DataCopyPad(outputGMY[progress], yLocal, extParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::ProcessChannel(int64_t n, int64_t c)
{
    // 算法：depthwise 卷积的标准形式。
    //   y[n,c,oh,ow] = b[c] + sum_{kh,kw} x[n,c,oh*stride-pad+kh*dil, ow*stride-pad+kw*dil] * w[c,0,kh,kw]
    // 仅当输入坐标落在 [0, H_in) × [0, W_in) 范围内时该项计入（越界视为 0）。
    //
    // 关键设计：
    //   - 每个任务 (n, c) 负责一个完整输出平面，任务间无数据依赖，天然可并行。
    //   - 逐输出行 oh 处理：对每个 kh 载入整行 x 到 UB（升 float），对每个 kw 计算
    //     其在输出行上的有效列范围，向量化累加到 outF。
    //   - stride==1 且块内有效列从 offset 0 开始时，用向量 Add（dst offset=0 可靠）；
    //     否则用标量 SetValue/GetValue 在 UB 内展开（纯 UB，不经 GM）。
    //   - stride>1 时输入列跨步，用标量 gather + 累加。
    //   - outF 全程以 offset 0 参与向量指令（规避真机 dst 运行期 offset 缺陷）。
    //   - 算完一行立即 DataCopyPad 写 GM（规避多核 SetValue 写 GM 丢失）。
    const int64_t xBase = (n * c_ + c) * planeIn_;
    const int64_t yBase = (n * c_ + c) * planeOut_;
    const float biasVal = (hasBias_ != 0) ? biasGMB.GetValue(static_cast<uint32_t>(c)) : 0.0f;

    LocalTensor<float> xF = xFBuf_.Get<float>();
    LocalTensor<float> xSeg = xSegBuf_.Get<float>();
    LocalTensor<float> outF = outFBuf_.Get<float>();
    LocalTensor<T> wLocal = wBuf_.Get<T>();

    // 加载 w[c, 0, :, :]（kH*kW 个连续元素）到 wBuf（MTE2→V 同步后 GetValue 读）
    {
        DataCopyExtParams wExt{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(kH_ * kW_ * static_cast<int64_t>(sizeof(T))),
            0,
            0,
            0};
        DataCopyPadExtParams<T> wPad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(wLocal, weightGMW[c * kH_ * kW_], wExt, wPad);
        // MTE2→V：wLocal 刚由 DataCopyPad 写入，后续 GetValue 读它。
        // 不用 HardEvent::MTE2_S 事件对（已过评测算子 0 处使用，平台上不可靠）。
        event_t eM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eM2V);
        WaitFlag<HardEvent::MTE2_V>(eM2V);
    }

    for (int64_t oh = 0; oh < hOut_; ++oh) {
        for (int64_t owStart = 0; owStart < wOut_;) {
            int64_t remain = wOut_ - owStart;
            int64_t owBlk = (remain < owBlkMax_) ? remain : owBlkMax_;
            int64_t owBlkAl = VecAlignCount<float>(owBlk);

            // 整段输出行的初始值 = 偏置
            Duplicate(outF, biasVal, static_cast<int32_t>(owBlkAl));

            for (int64_t kh = 0; kh < kH_; ++kh) {
                int64_t ih = oh * stride_ - pad_ + kh * dil_;
                if (ih < 0 || ih >= hIn_) {
                    continue;
                }

                // 载入整行 x[n,c,ih,0..W_in) 到 UB，升 float
                CopyIn(xBase + ih * wIn_, wIn_);
                LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
                CastOrCopy<T>(xF, xLocal, VecAlignCount<T>(wIn_));
                inputQueueX.FreeTensor(xLocal);

                for (int64_t kw = 0; kw < kW_; ++kw) {
                    float wv = static_cast<float>(wLocal.GetValue(static_cast<uint32_t>(kh * kW_ + kw)));
                    int64_t d = kw * dil_ - pad_;

                    // 全局有效 ow 范围：0 <= ow < W_out 且 0 <= ow*stride + d < W_in
                    //   gLo = max(0, ceil(-d/stride))
                    //   gHi = min(W_out, ceil((W_in-d)/stride))
                    int64_t gLo, gHi;
                    if (-d <= 0) {
                        gLo = 0;
                    } else {
                        gLo = (-d + stride_ - 1) / stride_;
                    }
                    int64_t num = wIn_ - d;
                    if (num <= 0) {
                        gHi = 0;
                    } else {
                        gHi = (num + stride_ - 1) / stride_;
                    }
                    if (gHi > wOut_) {
                        gHi = wOut_;
                    }
                    if (gLo < 0) {
                        gLo = 0;
                    }
                    if (gLo >= gHi) {
                        continue;
                    }

                    // 与块 [owStart, owStart+owBlk) 求交，得到块内有效范围 [blkLo, blkHi)
                    int64_t blkLo = (gLo > owStart) ? (gLo - owStart) : 0;
                    int64_t blkHi = (gHi < owStart + owBlk) ? (gHi - owStart) : owBlk;
                    if (blkLo >= blkHi) {
                        continue;
                    }
                    int64_t cnt = blkHi - blkLo;

                    if (stride_ == 1) {
                        // stride==1：输入列连续，从 GM 加载子段到 xF（基地址 32B 对齐）
                        int64_t xOff = xBase + ih * wIn_ + owStart + blkLo + d;
                        CopyIn(xOff, cnt);
                        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
                        CastOrCopy<T>(xF, xLocal, VecAlignCount<T>(cnt));
                        inputQueueX.FreeTensor(xLocal);

                        int64_t cntAl = VecAlignCount<float>(cnt);
                        // Axpy 向量化：xSeg = xF * wv（src=xF 基地址对齐，dst=xSeg 基地址对齐）
                        Duplicate(xSeg, 0.0f, static_cast<int32_t>(cntAl));
                        Axpy(xSeg, xF, wv, static_cast<int32_t>(cntAl));
                        // 标量展开：outF[blkLo+t] += xSeg[t]（纯 UB，不经 GM）
                        for (int64_t t = 0; t < cnt; ++t) {
                            outF.SetValue(static_cast<uint32_t>(blkLo + t),
                                          outF.GetValue(static_cast<uint32_t>(blkLo + t)) +
                                          xSeg.GetValue(static_cast<uint32_t>(t)));
                        }
                    } else {
                        // stride>1：输入列跨步，标量 gather + 累加
                        int64_t owGlobal = owStart + blkLo;
                        for (int64_t t = 0; t < cnt; ++t) {
                            int64_t ow = owGlobal + t;
                            int64_t iw = ow * stride_ + d;
                            outF.SetValue(static_cast<uint32_t>(blkLo + t),
                                          outF.GetValue(static_cast<uint32_t>(blkLo + t)) +
                                          xF.GetValue(static_cast<uint32_t>(iw)) * wv);
                        }
                    }
                }
            }

            Compute(owBlk);
            {
                // V→MTE3：outFBuf_ 刚由 Cast 写入，随后 DataCopyPad 读它写 GM
                event_t eV2M = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eV2M);
                WaitFlag<HardEvent::V_MTE3>(eV2M);
            }
            CopyOut(yBase + oh * wOut_ + owStart, owBlk);
            owStart += owBlk;
        }
    }
}

template <typename T>
__aicore__ inline void DepthwiseConv2D<T>::Process()
{
    // TODO: 实现 Process 逻辑
    for (int64_t task = taskStart_; task < taskEnd_; ++task) {
        int64_t n = task / c_;
        int64_t c = task - n * c_;
        ProcessChannel(n, c);
    }

    // 确保所有 MTE3（GM 写出）在 kernel 退出前完成
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
}

} // namespace NsDepthwiseConv2D
#endif // DEPTHWISECONV2D_H
