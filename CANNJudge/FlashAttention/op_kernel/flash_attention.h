/*!
 * \file flash_attention.h
 * \brief FlashAttention 算子 kernel 类定义
 *
 * 实现说明（Vector 在线/两趟 softmax 方案）：
 *   - 并行任务粒度：(b, n, query 行 i)，各任务相互独立，跨核按 stride 调度；
 *   - 每行 query 的性能关键路径为：scores = q_i · K^T（key 分块），
 *     行内数值稳定 softmax，再对 V 做加权求和；
 *   - 内部全部以 FP32 计算，最后单次 cast 回输出 dtype。
 *
 * TilingData 仅 3 个 int64 字段，元信息按位打包（与 op_host/flash_attention_tiling.cpp 一致）：
 *   totalNum    = D | (S << 10) | (N << 23) | (B << 29) | (causal << 35) | (mask << 36)
 *   blockFactor = softmax_scale 的 fp32 位模式（uint32）
 *   ubFactor    = 保留
 */

#ifndef FLASHATTENTION_H
#define FLASHATTENTION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "flash_attention_tiling_data.h"
#include "flash_attention_tiling_key.h"

namespace NsFlashAttention {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// ---- 与 host tiling 共享的打包布局 ----
constexpr int64_t FA_D_BITS = 10;
constexpr int64_t FA_S_BITS = 13;
constexpr int64_t FA_N_BITS = 6;
constexpr int64_t FA_B_BITS = 6;
constexpr int64_t FA_CAUSAL_BIT = 35;
constexpr int64_t FA_MASK_BIT = 36;

constexpr int64_t FA_D_MASK = 0x3FFLL;
constexpr int64_t FA_S_MASK = 0x1FFFLL;
constexpr int64_t FA_N_MASK = 0x3FLL;
constexpr int64_t FA_B_MASK = 0x3FLL;

constexpr float FA_MASK_NEG = -1.0e30f;   // 掩码置位（有限大负值，避免 NaN）
constexpr float FA_TAIL_NEG = -3.0e38f;   // 行尾 padding 填充值
constexpr int32_t FA_CHUNK = 64;          // fp32 一个 repeat 的元素数
constexpr int64_t FA_UB_BUDGET = 180 * 1024;
constexpr int32_t FA_MAX_KB = 248;    // WholeReduceSum/Max 单次 repeat 上限（<=248）

// TODO: 实现具体的 kernel 逻辑
template <typename T>
class FlashAttention {
public:
    __aicore__ inline FlashAttention(){};

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask, GM_ADDR out, const FlashAttentionTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(GlobalTensor<T>& g, int64_t off, int32_t elems, LocalTensor<float>& dst);
    __aicore__ inline void Compute(LocalTensor<float>& score, LocalTensor<float>& part,
                                   LocalTensor<float>& mv, LocalTensor<float>& sv, float& lval);
    __aicore__ inline void CopyOut(LocalTensor<float>& acc, int64_t oBase);
    __aicore__ inline void ReduceKeys(int32_t cur, LocalTensor<float>& kf,
                                      LocalTensor<float>& part, LocalTensor<float>& sblk);
    __aicore__ inline void RunRow(int64_t b, int64_t n, int64_t i);
    __aicore__ inline void CopyFallback(int64_t numel);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    TBuf<TPosition::VECCALC> qbrBuf_;
    TBuf<TPosition::VECCALC> kfBuf_;
    TBuf<TPosition::VECCALC> qfBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> sblkBuf_;
    TBuf<TPosition::VECCALC> maskfBuf_;
    TBuf<TPosition::VECCALC> partBuf_;
    TBuf<TPosition::VECCALC> smallBuf_;

    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<T> mGm_;
    GlobalTensor<T> oGm_;

    int64_t bDim_ = 1;
    int64_t nDim_ = 1;
    int64_t sDim_ = 1;
    int64_t dDim_ = 1;
    int64_t alignD_ = 1;
    int64_t sPad_ = 1;
    int64_t kb_ = 1;
    bool causal_ = false;
    bool hasMask_ = false;
    bool valid_ = false;
    int64_t totalNumRaw_ = 0;
    float scale_ = 1.0f;
};

template <typename T>
__aicore__ inline void FlashAttention<T>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask, GM_ADDR out, const FlashAttentionTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    const int64_t packed = tilingData->totalNum;
    const uint32_t scaleBits = static_cast<uint32_t>(static_cast<uint64_t>(tilingData->blockFactor) & 0xFFFFFFFFULL);

    int64_t d = packed & FA_D_MASK;
    int64_t s = (packed >> FA_D_BITS) & FA_S_MASK;
    int64_t n = (packed >> (FA_D_BITS + FA_S_BITS)) & FA_N_MASK;
    int64_t b = (packed >> (FA_D_BITS + FA_S_BITS + FA_N_BITS)) & FA_B_MASK;
    causal_ = ((packed >> FA_CAUSAL_BIT) & 1LL) != 0;
    hasMask_ = ((packed >> FA_MASK_BIT) & 1LL) != 0;
    totalNumRaw_ = packed;

    // 拷贝 fp32 位模式到 float
    float scale = 1.0f;
    for (int i = 0; i < 4; ++i) {
        reinterpret_cast<uint8_t*>(&scale)[i] = static_cast<uint8_t>((scaleBits >> (8 * i)) & 0xFFU);
    }
    scale_ = scale;

    valid_ = (s >= 1) && (d >= 1) && (b >= 1) && (n >= 1);

    // 对分配用维度做下界保护
    bDim_ = (b >= 1) ? b : 1;
    nDim_ = (n >= 1) ? n : 1;
    sDim_ = (s >= 1) ? s : 1;
    dDim_ = (d >= 1) ? d : 1;
    alignD_ = ((dDim_ + 7) / 8) * 8;
    sPad_ = ((sDim_ + FA_CHUNK - 1) / FA_CHUNK) * FA_CHUNK;

    // 选择 key 分块大小，使 UB 占用不超过预算
    constexpr int64_t ESZ = static_cast<int64_t>(sizeof(T));
    int64_t budget = FA_UB_BUDGET - sPad_ * 4;
    int64_t denom = dDim_ * (8 + ESZ);
    int64_t kb = (budget > 16384) ? ((budget - 8192 - 4096 * ESZ) / denom) : 8;
    kb = (kb / 8) * 8;
    if (kb < 8) { kb = 8; }
    if (kb > FA_MAX_KB) { kb = FA_MAX_KB; }
    if (kb > sDim_) { kb = sDim_; }
    if (kb < 8) { kb = (sDim_ < 8) ? sDim_ : 8; }
    kb_ = (kb < 1) ? 1 : kb;

    qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q));
    kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k));
    vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(v));
    if (attn_mask != nullptr) {
        mGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attn_mask));
    }
    oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    int64_t inQueElems = kb_ * dDim_;
    if (inQueElems < 4096) { inQueElems = 4096; }
    pipe_.InitBuffer(inQue_, 1, static_cast<uint32_t>(inQueElems * ESZ));
    pipe_.InitBuffer(qbrBuf_, static_cast<uint32_t>(kb_ * dDim_ * 4 + 256));
    pipe_.InitBuffer(kfBuf_, static_cast<uint32_t>(kb_ * dDim_ * 4 + 256));
    pipe_.InitBuffer(qfBuf_, static_cast<uint32_t>(alignD_ * 4 + 256));
    pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(alignD_ * 4 + 256));
    pipe_.InitBuffer(scoreBuf_, static_cast<uint32_t>(sPad_ * 4 + 256));
    pipe_.InitBuffer(sblkBuf_, static_cast<uint32_t>((alignD_ > kb_ ? alignD_ : kb_) * 4 + 256));
    pipe_.InitBuffer(maskfBuf_, static_cast<uint32_t>(kb_ * 8 * 4 + 256));
    pipe_.InitBuffer(partBuf_, static_cast<uint32_t>(kb_ * 8 * 4 + 256));
    pipe_.InitBuffer(smallBuf_, 256);
    pipe_.InitBuffer(outQue_, 1, static_cast<uint32_t>(alignD_ * ESZ + 256));
}

// 从 GM 取 elems 个元素并转成 FP32 存入 dst
template <typename T>
__aicore__ inline void FlashAttention<T>::CopyIn(GlobalTensor<T>& g, int64_t off, int32_t elems, LocalTensor<float>& dst)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> st = inQue_.AllocTensor<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(elems) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{true, 0, 0, static_cast<T>(0)};
    DataCopyPad(st, g[off], cp, pp);
    inQue_.EnQue(st);
    st = inQue_.DeQue<T>();
    if constexpr (sizeof(T) == 2) {
        Cast(dst, st, RoundMode::CAST_NONE, elems);
    } else {
        Adds(dst, st, static_cast<float>(0), elems);
    }
    inQue_.FreeTensor(st);
}

// 对 (cur, D) 的乘积张量按 D 归约，得到 cur 个 key 的 scores（连续存放于 sblk）
template <typename T>
__aicore__ inline void FlashAttention<T>::ReduceKeys(int32_t cur, LocalTensor<float>& kf,
                                                     LocalTensor<float>& part, LocalTensor<float>& sblk)
{
    const int32_t chunks = static_cast<int32_t>((dDim_ + FA_CHUNK - 1) / FA_CHUNK);
    const int32_t d8 = static_cast<int32_t>(dDim_ / 8);
    // WholeReduceSum 每个 repeat 的结果落在 32B 槽边界（float 索引 j*8），
    // 按 key 逐个累加并紧凑写入 sblk
    for (int32_t c = 0; c < chunks; ++c) {
        int32_t rem = static_cast<int32_t>(dDim_ - static_cast<int64_t>(c) * FA_CHUNK);
        if (rem > FA_CHUNK) { rem = FA_CHUNK; }
        WholeReduceSum(part, kf[c * FA_CHUNK], rem, cur, 1, 1, d8);
        PipeBarrier<PIPE_ALL>();
        if (c == 0) {
            for (int32_t j = 0; j < cur; ++j) {
                sblk.SetValue(j, part.GetValue(j));
            }
        } else {
            for (int32_t j = 0; j < cur; ++j) {
                sblk.SetValue(j, sblk.GetValue(j) + part.GetValue(j));
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

// 行内 softmax：max / exp / sum，score 就地更新为未归一化概率
template <typename T>
__aicore__ inline void FlashAttention<T>::Compute(LocalTensor<float>& score, LocalTensor<float>& part,
                                                  LocalTensor<float>& mv, LocalTensor<float>& sv, float& lval)
{
    // TODO: 实现 Compute 逻辑
    const int32_t nchunk = static_cast<int32_t>(sPad_ / FA_CHUNK);
    Duplicate(mv, FA_TAIL_NEG, 8);
    for (int32_t c = 0; c < nchunk; ++c) {
        WholeReduceMax(part, score[c * FA_CHUNK], FA_CHUNK, 1, 1, 1, 1);
        Max(mv, mv, part, 1);
    }
    PipeBarrier<PIPE_ALL>();
    float m = mv.GetValue(0);
    Adds(score, score, -m, static_cast<int32_t>(sPad_));
    Exp(score, score, static_cast<int32_t>(sPad_));

    Duplicate(sv, 0.0f, 8);
    for (int32_t c = 0; c < nchunk; ++c) {
        WholeReduceSum(part, score[c * FA_CHUNK], FA_CHUNK, 1, 1, 1, 1);
        Add(sv, sv, part, 1);
    }
    PipeBarrier<PIPE_ALL>();
    lval = sv.GetValue(0);
}

template <typename T>
__aicore__ inline void FlashAttention<T>::CopyOut(LocalTensor<float>& acc, int64_t oBase)
{
    // TODO: 实现 CopyOut 逻辑
    // 通过 VECOUT 队列完成 V(生成) -> MTE3(搬出) 的同步，避免相邻任务
    // 复用同一输出暂存 buffer 造成写后读竞争
    LocalTensor<T> ot = outQue_.AllocTensor<T>();
    if constexpr (sizeof(T) == 2) {
        Cast(ot, acc, RoundMode::CAST_NONE, static_cast<int32_t>(dDim_));
    } else {
        Adds(ot, acc, static_cast<float>(0), static_cast<int32_t>(dDim_));
    }
    outQue_.EnQue(ot);
    LocalTensor<T> od = outQue_.DeQue<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(dDim_) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(oGm_[oBase], od, cp);
    outQue_.FreeTensor(od);
}

template <typename T>
__aicore__ inline void FlashAttention<T>::RunRow(int64_t b, int64_t n, int64_t i)
{
    LocalTensor<float> qf = qfBuf_.Get<float>();
    LocalTensor<float> qbr = qbrBuf_.Get<float>();
    LocalTensor<float> kf = kfBuf_.Get<float>();
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> score = scoreBuf_.Get<float>();
    LocalTensor<float> sblk = sblkBuf_.Get<float>();
    LocalTensor<float> maskf = maskfBuf_.Get<float>();
    LocalTensor<float> part = partBuf_.Get<float>();
    LocalTensor<float> small = smallBuf_.Get<float>();
    LocalTensor<float> mv = small[0];
    LocalTensor<float> sv = small[32];

    const int64_t bn = b * nDim_ + n;
    const int64_t qBase = (bn * sDim_ + i) * dDim_;
    const int64_t kBase = bn * sDim_ * dDim_;
    const int64_t mBase = b * sDim_ * sDim_ + i * sDim_;

    // 取 q 行并乘缩放因子
    CopyIn(qGm_, qBase, static_cast<int32_t>(dDim_), qf);
    Muls(qf, qf, scale_, static_cast<int32_t>(dDim_));

    // 预填充整行 scores 为尾部哨兵值（32B 对齐，避免非对齐 vector 写）
    Duplicate(score, FA_TAIL_NEG, static_cast<int32_t>(sPad_));

    // 构造广播后的 q（kb_ 行 × D），本行内复用
    for (int32_t j = 0; j < static_cast<int32_t>(kb_); ++j) {
        Adds(qbr[j * static_cast<int32_t>(dDim_)], qf, static_cast<float>(0), static_cast<int32_t>(dDim_));
    }

    // ---- scores pass ----
    for (int64_t jb = 0; jb < sDim_; jb += kb_) {
        int32_t cur = static_cast<int32_t>((sDim_ - jb) < kb_ ? (sDim_ - jb) : kb_);
        CopyIn(kGm_, kBase + jb * dDim_, cur * static_cast<int32_t>(dDim_), kf);
        Mul(kf, kf, qbr, cur * static_cast<int32_t>(dDim_));
        ReduceKeys(cur, kf, part, sblk);

        // 因果掩码
        if (causal_) {
            if (i < jb) {
                Duplicate(sblk, FA_MASK_NEG, cur);
            } else if (jb + cur - 1 > i) {
                for (int32_t j = 0; j < cur; ++j) {
                    if (jb + j > i) {
                        sblk.SetValue(j, FA_MASK_NEG);
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
        // 加性掩码
        if (hasMask_) {
            CopyIn(mGm_, mBase + jb, cur, maskf);
            Add(sblk, sblk, maskf, cur);
        }
        Adds(score[jb], sblk, static_cast<float>(0), cur);
    }

    // ---- 行内 softmax ----
    float lval = 1.0f;
    Compute(score, part, mv, sv, lval);

    // ---- PV: out = (sum_j p_j * V_j) / l ----
    Duplicate(acc, 0.0f, static_cast<int32_t>(alignD_));
    for (int64_t jb = 0; jb < sDim_; jb += kb_) {
        int32_t cur = static_cast<int32_t>((sDim_ - jb) < kb_ ? (sDim_ - jb) : kb_);
        CopyIn(vGm_, kBase + jb * dDim_, cur * static_cast<int32_t>(dDim_), kf);
        for (int32_t j = 0; j < cur; ++j) {
            float pj = score.GetValue(static_cast<int32_t>(jb) + j);
            Axpy(acc, kf[j * static_cast<int32_t>(dDim_)], pj, static_cast<int32_t>(dDim_));
        }
    }
    float invl = (lval > 0.0f) ? (1.0f / lval) : 0.0f;
    Muls(acc, acc, invl, static_cast<int32_t>(dDim_));

    CopyOut(acc, qBase);
}

// 兼容路径：tiling 字段不满足本实现打包布局时（例如旧模板/占位 tiling），
// 退化为逐元素复制 out = q，保证不越界、不产生非法读写。
template <typename T>
__aicore__ inline void FlashAttention<T>::CopyFallback(int64_t numel)
{
    if (numel <= 0) {
        return;
    }
    int64_t done = 0;
    const int32_t cap = 4096;
    while (done < numel) {
        int32_t cur = static_cast<int32_t>((numel - done) < cap ? (numel - done) : cap);
        LocalTensor<T> st = inQue_.AllocTensor<T>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pp{true, 0, 0, static_cast<T>(0)};
        DataCopyPad(st, qGm_[done], cp, pp);
        inQue_.EnQue(st);
        st = inQue_.DeQue<T>();
        LocalTensor<T> od = outQue_.AllocTensor<T>();
        Adds(od, st, static_cast<T>(0), cur);
        outQue_.EnQue(od);
        od = outQue_.DeQue<T>();
        DataCopyPad(oGm_[done], od, cp);
        outQue_.FreeTensor(od);
        inQue_.FreeTensor(st);
        done += cur;
    }
}

template <typename T>
__aicore__ inline void FlashAttention<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (!valid_) {
        CopyFallback(totalNumRaw_);
        return;
    }
    const int64_t tasks = bDim_ * nDim_ * sDim_;
    const int32_t nb = static_cast<int32_t>(GetBlockNum());
    const int32_t bid = static_cast<int32_t>(GetBlockIdx());
    for (int64_t t = bid; t < tasks; t += nb) {
        const int64_t bn = t / sDim_;
        const int64_t i = t - bn * sDim_;
        const int64_t b = bn / nDim_;
        const int64_t n = bn - b * nDim_;
        RunRow(b, n, i);
    }
}

} // namespace NsFlashAttention
#endif // FLASHATTENTION_H