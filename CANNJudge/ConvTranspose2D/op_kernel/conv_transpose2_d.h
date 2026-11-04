/*!
 * \file conv_transpose2_d.h
 * \brief ConvTranspose2D 算子 kernel 类定义
 */

#ifndef CONVTRANSPOSE2D_H
#define CONVTRANSPOSE2D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "conv_transpose2_d_tiling_data.h"
#include "conv_transpose2_d_tiling_key.h"

#include <type_traits>

namespace NsConvTranspose2D {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// UB 内单块处理的上限（编译期固定，保证任何合法 shape 下 UB 都不溢出）。
//   TR_MAX  ：单个残差类 r 下一次处理的输出列数 t 的上限；
//   owBlkMax ：TR_MAX * stride，即一次处理的连续输出列数上限；
//   CIN_MAX  ：wSeg（沿 C_in 的权重段）上限。
// 三者与题面维度上界（H/W ≤ 1024、k ≤ 15、C ≤ 4096）共同决定 UB 峰值占用。
constexpr int64_t TR_MAX = 4096;
constexpr int64_t CIN_MAX = 8192;
constexpr int64_t kMaxK = 16;   // 题面 kH/kW <= 15，留 1 余量

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

// TilingData 位域约定（与 op_host/conv_transpose2_d_tiling.cpp 严格一致）。
// 刻意不把任何关键标量放在 totalNum：真实框架下 TilingData 首字段在 kernel 侧读取不可靠
// （偶发读到 0 / 8，已由常量 tiling 受控实验确证），故 kernel 完全不读 totalNum。
//   blockFactor : bit[0:13) = N,  bit[13:26) = C_in, bit[26:39) = C_out,
//                 bit[39:50) = H_in, bit[50:61) = W_in, bit[61:64) 保留
//   ubFactor    : bit[0:4) = kH, bit[4:8) = kW, bit[8:20) = stride,
//                 bit[20:32) = padding, bit[32:44) = output_padding,
//                 bit[44:56) = dilation, bit[56:57) = hasBias, bit[57:64) 保留
constexpr uint64_t BF_N_SHIFT = 0;
constexpr uint64_t BF_CIN_SHIFT = 13;
constexpr uint64_t BF_COUT_SHIFT = 26;
constexpr uint64_t BF_HIN_SHIFT = 39;
constexpr uint64_t BF_WIN_SHIFT = 50;
constexpr uint64_t BF_N_MASK = 0x1FFFULL;
constexpr uint64_t BF_CIN_MASK = 0x1FFFULL;
constexpr uint64_t BF_COUT_MASK = 0x1FFFULL;
constexpr uint64_t BF_HIN_MASK = 0x7FFULL;
constexpr uint64_t BF_WIN_MASK = 0x7FFULL;

constexpr uint64_t UF_KH_SHIFT = 0;
constexpr uint64_t UF_KW_SHIFT = 4;
constexpr uint64_t UF_STRIDE_SHIFT = 8;
constexpr uint64_t UF_PAD_SHIFT = 20;
constexpr uint64_t UF_OPAD_SHIFT = 32;
constexpr uint64_t UF_DIL_SHIFT = 44;
constexpr uint64_t UF_HASBIAS_SHIFT = 56;
constexpr uint64_t UF_K_MASK = 0xFULL;
constexpr uint64_t UF_PARAM_MASK = 0xFFFULL;

template <typename T>
class ConvTranspose2D {
public:
    __aicore__ inline ConvTranspose2D(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, const ConvTranspose2DTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

    // 单个 (n, co) 输出平面的完整计算。
    __aicore__ inline void ProcessChannel(int64_t n, int64_t co);

    // 取回 w[:, co, kh, kW] 这一段（沿 C_in 跨步 wPlane_）到 wSegBuf_
    __aicore__ inline void LoadWeightSeg(int64_t co, int64_t kh, int64_t kw);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // 计算用 UB。accBuf_ 全程只以 offset 0 参与向量指令，规避真机上
    // "dst 带运行期 offset 的向量指令只有 offset=0 生效"的缺陷。
    TBuf<TPosition::VECCALC> accBuf_;    // float 累加器 [trBlkAl]
    TBuf<TPosition::VECCALC> xFBuf_;     // x 行升 float 后的缓冲 [trBlkAl]
    TBuf<TPosition::VECCALC> outFBuf_;   // 连续输出列的 float 缓冲 [owBlkAl]
    TBuf<TPosition::VECCALC> wSegBuf_;   // w[:, co, kh, kw] 段（T 类型）[cIn]

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> weightGMW;
    GlobalTensor<float> biasGMB;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;

    int64_t n_ = 1;
    int64_t cIn_ = 1;
    int64_t cOut_ = 1;
    int64_t hIn_ = 1;
    int64_t wIn_ = 1;
    int64_t kH_ = 1;
    int64_t kW_ = 1;
    int64_t stride_ = 1;
    int64_t padding_ = 0;
    int64_t outputPadding_ = 0;
    int64_t dilation_ = 1;
    int64_t hasBias_ = 0;

    int64_t hOut_ = 1;
    int64_t wOut_ = 1;
    int64_t planeIn_ = 1;    // H_in * W_in
    int64_t planeOut_ = 1;   // H_out * W_out
    int64_t wPlane_ = 1;     // C_out * kH * kW（w 中沿 C_in 的 stride）
    int64_t totalTask_ = 1;  // N * C_out

    int64_t taskStart_ = 0;
    int64_t taskEnd_ = 0;

    int64_t trMax_ = 1;      // 实际分配的 TrBlk 上限
    bool useWSeg_ = true;    // C_in 超过 CIN_MAX 时回退为逐元素 GM 读 w
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void ConvTranspose2D<T>::Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, const ConvTranspose2DTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 1. 解包 TilingData（位域与 op_host tiling.cpp 一致；不读 totalNum）
    uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);

    n_ = static_cast<int64_t>((blockFactor >> BF_N_SHIFT) & BF_N_MASK);
    cIn_ = static_cast<int64_t>((blockFactor >> BF_CIN_SHIFT) & BF_CIN_MASK);
    cOut_ = static_cast<int64_t>((blockFactor >> BF_COUT_SHIFT) & BF_COUT_MASK);
    hIn_ = static_cast<int64_t>((blockFactor >> BF_HIN_SHIFT) & BF_HIN_MASK);
    wIn_ = static_cast<int64_t>((blockFactor >> BF_WIN_SHIFT) & BF_WIN_MASK);

    kH_ = static_cast<int64_t>((ubFactor >> UF_KH_SHIFT) & UF_K_MASK);
    kW_ = static_cast<int64_t>((ubFactor >> UF_KW_SHIFT) & UF_K_MASK);
    stride_ = static_cast<int64_t>((ubFactor >> UF_STRIDE_SHIFT) & UF_PARAM_MASK);
    padding_ = static_cast<int64_t>((ubFactor >> UF_PAD_SHIFT) & UF_PARAM_MASK);
    outputPadding_ = static_cast<int64_t>((ubFactor >> UF_OPAD_SHIFT) & UF_PARAM_MASK);
    dilation_ = static_cast<int64_t>((ubFactor >> UF_DIL_SHIFT) & UF_PARAM_MASK);
    hasBias_ = static_cast<int64_t>((ubFactor >> UF_HASBIAS_SHIFT) & 0x1ULL);

    // 占位 / 退化 tiling 的一致性保护（本地 UT 用占位值，须保证不除零、不死循环、不越界）
    if (n_ < 1) { n_ = 1; }
    if (cIn_ < 1) { cIn_ = 1; }
    if (cOut_ < 1) { cOut_ = 1; }
    if (hIn_ < 1) { hIn_ = 1; }
    if (wIn_ < 1) { wIn_ = 1; }
    if (kH_ < 1) { kH_ = 1; }
    if (kW_ < 1) { kW_ = 1; }
    if (stride_ < 1) { stride_ = 1; }
    if (dilation_ < 1) { dilation_ = 1; }
    if (padding_ < 0) { padding_ = 0; }
    if (outputPadding_ < 0) { outputPadding_ = 0; }

    // H_out = (H_in - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1
    hOut_ = (hIn_ - 1) * stride_ - 2 * padding_ + dilation_ * (kH_ - 1) + outputPadding_ + 1;
    wOut_ = (wIn_ - 1) * stride_ - 2 * padding_ + dilation_ * (kW_ - 1) + outputPadding_ + 1;
    if (hOut_ < 1) { hOut_ = 1; }
    if (wOut_ < 1) { wOut_ = 1; }

    planeIn_ = hIn_ * wIn_;
    planeOut_ = hOut_ * wOut_;
    wPlane_ = cOut_ * kH_ * kW_;
    totalTask_ = n_ * cOut_;
    useWSeg_ = (cIn_ <= CIN_MAX);

    // 2. GlobalTensor 绑定
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint32_t>(n_ * cIn_ * planeIn_));
    weightGMW.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(w), static_cast<uint32_t>(cIn_ * wPlane_));
    biasGMB.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint32_t>(cOut_));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), static_cast<uint32_t>(n_ * cOut_ * planeOut_));

    // 3. 多核切分：任务 = (n, co) 输出平面，任务间无数据依赖，按任务均分、无跨核竞争
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

    // 4. UB 分配：单块 TrBlk 上限与 UB 容量共同决定，峰值占用远低于 192KB 池上限
    trMax_ = (wOut_ + stride_ - 1) / stride_;
    if (trMax_ > TR_MAX) { trMax_ = TR_MAX; }
    if (trMax_ < 1) { trMax_ = 1; }
    int64_t owBlkMax = trMax_ * stride_;

    // 向量指令（Cast/Axpy/Duplicate）在 CPU 调试构建下要求 dst/src 张量至少 64 字节
    // （CheckFunVecBinaryScalarDiffType），小 shape（wOut 很小）时须把各子区抬到该下限，
    // 否则会 abort。代价至多每块 64B，对峰值占用无影响。
    constexpr int64_t kMinVecElems = 64 / static_cast<int64_t>(sizeof(T));
    int64_t trBlkAl = VecAlignCount<float>(trMax_);
    if (trBlkAl < kMinVecElems) { trBlkAl = kMinVecElems; }
    int64_t owBlkAl = VecAlignCount<T>(owBlkMax);
    if (owBlkAl < kMinVecElems) { owBlkAl = kMinVecElems; }
    // wSeg 每行只取 1 个元素（blockLen = sizeof(T) 字节），但 DataCopyPad 的尺寸检查器
    // 按 blockCount * AlignUp(blockLen, 32) 计费——即每行至少占一整个 32B 块。
    // 故 wSegBuf_ 须按「行数 × 每行 32B 可容纳元素数」分配，否则越界。
    constexpr int64_t kWSegRowElems = 32 / static_cast<int64_t>(sizeof(T));
    int64_t wSegElems = useWSeg_ ? (cIn_ * kWSegRowElems) : kWSegRowElems;
    if (wSegElems < kMinVecElems) { wSegElems = kMinVecElems; }

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(trBlkAl * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(owBlkAl * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(accBuf_, static_cast<uint32_t>(trBlkAl * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(xFBuf_, static_cast<uint32_t>(trBlkAl * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(outFBuf_, static_cast<uint32_t>(owBlkAl * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(wSegBuf_, static_cast<uint32_t>(wSegElems * static_cast<int64_t>(sizeof(T))));
    ubLength_ = trBlkAl;
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
// 尾部不足 32B 的部分由 DataCopyPad 的 rightPadding 补 0，使后续 Cast / Axpy 的 count
// 可以安全取 32B 对齐值（pad 元素对累加结果贡献 0，不影响正确性）。
template <typename T>
__aicore__ inline void ConvTranspose2D<T>::CopyIn(int64_t progress, int64_t currentNum)
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

// Compute：把 outFBuf_ 中已重排成连续输出列的 float 结果转换为输出 dtype，送入 outputQueueY。
template <typename T>
__aicore__ inline void ConvTranspose2D<T>::Compute(int64_t currentNum)
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

// CopyOut：把 outputQueueY 中的整段连续输出列写出到 y（MTE3）。
// 全程连续写 GM，不用 GlobalTensor::SetValue——多核各自标量写 GM 会大面积丢失（已确证教训）。
template <typename T>
__aicore__ inline void ConvTranspose2D<T>::CopyOut(int64_t progress, int64_t currentNum)
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

// LoadWeightSeg：取回 w[:, co, kh, kw]（沿 C_in 跨步 wPlane_）到 wSegBuf_。
// DataCopyPad 每行 blockLen = sizeof(T) 字节，但硬件按 32B 块粒度排布，故第 ci 行的
// 有效元素位于 wSeg[ci * (32/sizeof(T))]。
template <typename T>
__aicore__ inline void ConvTranspose2D<T>::LoadWeightSeg(int64_t co, int64_t kh, int64_t kw)
{
    // TODO: 实现 LoadWeightSeg 逻辑
    if (!useWSeg_) {
        return;
    }
    LocalTensor<T> wSeg = wSegBuf_.Get<T>();
    DataCopyExtParams wExt{
        static_cast<uint16_t>(cIn_),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>((wPlane_ - 1) * static_cast<int64_t>(sizeof(T))),
        0,
        0};
    DataCopyPadExtParams<T> wPad{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(wSeg, weightGMW[co * kH_ * kW_ + kh * kW_ + kw], wExt, wPad);
    // MTE2→V：wSeg 刚由 DataCopyPad 写入，后续 GetValue 读它。
    // 不用 HardEvent::MTE2_S 事件对（已过评测算子 0 处使用，平台上不可靠）。
    event_t eM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eM2V);
    WaitFlag<HardEvent::MTE2_V>(eM2V);
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::ProcessChannel(int64_t n, int64_t co)
{
    // 算法：转置卷积的聚合（gather）形式。
    //   y[n,co,oh,ow] = b[co] + sum_{kh,kw,ci} x[n,ci,ih,iw] * w[ci,co,kh,kw]
    //   ih = (oh + padding - kh*dilation) / stride
    //   iw = (ow + padding - kw*dilation) / stride
    // 仅当 ih、iw 均为整数且落在输入范围内时该项计入。
    //
    // 关键重排：kw 提到最外层。对固定 (oh, kh, kw)，能使 iw 为整数的 ow 构成唯一残差类
    //   r_kw = (kw*dilation - padding) mod stride，  ow = owLo + t*stride，
    // 其中 owLo = max(r_kw, kw*dilation - padding) 是该 kw 的首个有效输出列，
    // 对应 iw = iw0 + t，iw0 = (owLo + padding - kw*dilation) / stride >= 0。
    // 于是「输入列 iw」与「输出列 ow」双双等差连续：
    //   - 输入侧：x[n,ci,ih,iw0..iw0+cnt) 是一段连续内存，单条 DataCopyPad 取回；
    //   - 计算侧：每个 (kh,kw,ci) 一次 Axpy 外积（acc += xRow * wScalar），全程向量化；
    //   - 输出侧：acc 的 offset 恒为 0（规避真机 dst 运行期 offset 缺陷），
    //             算完立即标量展开进 outFBuf_（连续输出列 float 缓冲），
    //             最后整段一次性 DataCopyPad 写 GM（规避多核 SetValue 写 GM 丢失）。
    const int64_t xNBase = n * cIn_ * planeIn_;
    const int64_t yNBase = (n * cOut_ + co) * planeOut_;
    const float biasVal = (hasBias_ != 0) ? biasGMB.GetValue(static_cast<uint32_t>(co)) : 0.0f;

    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> xF = xFBuf_.Get<float>();
    LocalTensor<float> outF = outFBuf_.Get<float>();
    LocalTensor<T> wSeg = wSegBuf_.Get<T>();

    // 每个 kw 的首个有效输出列 owLo 与起始输入列 iw0（只依赖 kw，与 owStart 无关）
    // 预计算一次，避免在 oh / 块循环里重复取模
    int64_t kwOwLo[kMaxK];
    int64_t kwIw0[kMaxK];
    int64_t kwCntMax[kMaxK];
    for (int64_t kw = 0; kw < kW_; ++kw) {
        int64_t r = ((kw * dilation_ - padding_) % stride_ + stride_) % stride_;
        int64_t owLo = r;
        if (kw * dilation_ - padding_ > owLo) { owLo = kw * dilation_ - padding_; }
        kwOwLo[kw] = owLo;
        kwIw0[kw] = (owLo + padding_ - kw * dilation_) / stride_;
        kwCntMax[kw] = wIn_ - kwIw0[kw];
    }

    for (int64_t oh = 0; oh < hOut_; ++oh) {
        for (int64_t owStart = 0; owStart < wOut_;) {
            int64_t remain = wOut_ - owStart;
            int64_t trBlk = (remain + stride_ - 1) / stride_;
            if (trBlk > trMax_) { trBlk = trMax_; }
            if (trBlk < 1) { trBlk = 1; }
            int64_t owBlk = trBlk * stride_;
            if (owBlk > remain) { owBlk = remain; }
            int64_t owBlkAl = VecAlignCount<T>(owBlk);

            // 整段连续输出列的初始值 = 偏置。output_padding / padding 边界等无输入
            // 贡献的位置在此自然获得偏置，与参考实现一致。
            Duplicate(outF, biasVal, static_cast<int32_t>(owBlkAl));

            for (int64_t kw = 0; kw < kW_; ++kw) {
                int64_t owLo = kwOwLo[kw];
                if (owLo >= owStart + owBlk) { continue; }
                int64_t iw0 = kwIw0[kw];
                if (iw0 >= wIn_) { continue; }
                // owLo ≡ r (mod stride)，且 owStart 恒为 stride 的整数倍（owStart 累加的是
                // owBlk = trBlk * stride），故残差类内首个不早于 owStart 的输出列为
                //   first = max(owStart + r, owLo)。
                // owLo < owStart 时必须先跳到块内首列：否则 owOff 为负（越界写 UB），
                // 且输入列起点也未随之前移。
                int64_t rKw = owLo % stride_;
                int64_t first = owStart + rKw;
                if (owLo > first) { first = owLo; }
                if (first >= owStart + owBlk) { continue; }
                int64_t iwShift = (first - owLo) / stride_;
                if (iw0 + iwShift >= wIn_) { continue; }
                // 该残差类在本块内的有效个数（上界同时受输入列范围约束）
                int64_t cnt = (owStart + owBlk - first + stride_ - 1) / stride_;
                if (cnt > trBlk) { cnt = trBlk; }
                int64_t cntMax = kwCntMax[kw] - iwShift;
                if (cnt > cntMax) { cnt = cntMax; }
                if (cnt <= 0) { continue; }
                int64_t cntAl2 = VecAlignCount<float>(cnt);
                // acc[t] 对应块内输出列 first + t*stride - owStart
                int64_t owOff = first - owStart;
                iw0 += iwShift;

                for (int64_t kh = 0; kh < kH_; ++kh) {
                    int64_t nh = oh + padding_ - kh * dilation_;
                    if (nh < 0 || nh % stride_ != 0) { continue; }
                    int64_t ih = nh / stride_;
                    if (ih >= hIn_) { continue; }

                    // acc 在每个 kh 段开始清零，ci 循环在其上累加，展开后即弃
                    Duplicate(acc, 0.0f, static_cast<int32_t>(cntAl2));
                    LoadWeightSeg(co, kh, kw);

                    for (int64_t ci = 0; ci < cIn_; ++ci) {
                        float wv;
                        if (useWSeg_) {
                            wv = static_cast<float>(wSeg.GetValue(static_cast<uint32_t>(
                                ci * (32 / static_cast<int64_t>(sizeof(T))))));
                        } else {
                            wv = static_cast<float>(weightGMW.GetValue(static_cast<uint32_t>(
                                (ci * cOut_ + co) * kH_ * kW_ + kh * kW_ + kw)));
                        }
                        int64_t xOff = xNBase + (ci * hIn_ + ih) * wIn_ + iw0;
                        CopyIn(xOff, cnt);
                        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
                        CastOrCopy<T>(xF, xLocal, VecAlignCount<T>(cnt));
                        inputQueueX.FreeTensor(xLocal);
                        // 外积累加：acc[t] += x[iw0 + t] * wv（acc offset 恒为 0）
                        Axpy(acc, xF, wv, cntAl2);
                    }

                    // 标量展开：outF[owOff + t*stride] += acc[t]（纯 UB，不经 GM）
                    for (int64_t t = 0; t < cnt; ++t) {
                        outF.SetValue(static_cast<uint32_t>(owOff + t * stride_),
                                      outF.GetValue(static_cast<uint32_t>(owOff + t * stride_)) +
                                          acc.GetValue(static_cast<uint32_t>(t)));
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
            CopyOut(yNBase + oh * wOut_ + owStart, owBlk);
            owStart += owBlk;
        }
    }
}

template <typename T>
__aicore__ inline void ConvTranspose2D<T>::Process()
{
    // TODO: 实现 Process 逻辑
    for (int64_t task = taskStart_; task < taskEnd_; ++task) {
        int64_t n = task / cOut_;
        int64_t co = task - n * cOut_;
        ProcessChannel(n, co);
    }

    // 确保所有 MTE3（GM 写出）在 kernel 退出前完成
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
}

} // namespace NsConvTranspose2D
#endif // CONVTRANSPOSE2D_H
