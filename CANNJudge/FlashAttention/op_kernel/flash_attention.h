/*!
 * \file flash_attention.h
 * \brief FlashAttention 算子 kernel 类定义
 *
 * 实现说明（Vector 全向量化方案，消除逐元素标量搬运）：
 *   - 并行任务粒度：(b, n, query 行 i)，各任务相互独立，跨核按 stride 调度；
 *   - 每行 query 的性能关键路径为：scores = q_i · K^T（key 分块），
 *     行内数值稳定 softmax，再对 V 做加权求和；
 *   - scores 归约使用 WholeReduceSum（沿 D 方向分块归约，结果紧凑排布），
 *     softmax 的 max/sum 亦用向量 reduce；PV 使用 Brcb 行广播 + 行点乘 +
 *     行方向树形归约，全程无标量 GetValue/SetValue 内循环；
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

constexpr float FA_PAD_NEG = -3.0e38f;    // 行尾 / 被屏蔽位置的填充哨兵（exp 后为 0，无 NaN）
constexpr float FA_TAIL_NEG = -3.0e38f;   // softmax 归约初始值（与哨兵一致）
constexpr int32_t FA_CHUNK = 64;          // fp32 一个 repeat 的元素数
constexpr int64_t FA_UB_BUDGET = 160 * 1024;   // UB 实际 192KB，留足余量防分配失败
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
    __aicore__ inline void BuildQbr(LocalTensor<float>& qf, LocalTensor<float>& qbr, int32_t rows);
    __aicore__ inline void ReduceScores(int32_t cur, LocalTensor<float>& kf,
                                        LocalTensor<float>& part, LocalTensor<float>& sc, LocalTensor<float> dst);
    __aicore__ inline void RowMulBroadcast(LocalTensor<float>& vf, LocalTensor<float>& pbr, int32_t rows);
    __aicore__ inline void ReduceRowsToAcc(LocalTensor<float>& work, int32_t rows, LocalTensor<float>& acc);
    __aicore__ inline void CopyOut(LocalTensor<float>& acc, int64_t oBase);
    __aicore__ inline void RunRow(int64_t b, int64_t n, int64_t i);
    __aicore__ inline void CopyFallback(int64_t numel);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    TBuf<TPosition::VECCALC> qfBuf_;
    TBuf<TPosition::VECCALC> qbrBuf_;
    TBuf<TPosition::VECCALC> kfBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> scBuf_;
    TBuf<TPosition::VECCALC> partBuf_;
    TBuf<TPosition::VECCALC> pbrBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
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

    // 选择 key 分块大小 KB，使 UB 占用不超过预算
    constexpr int64_t ESZ = static_cast<int64_t>(sizeof(T));
    // 固定占用：score + acc + qf + outQue + small + 12 个 buffer 各 256B 对齐 padding
    int64_t fixed = sPad_ * 4 + alignD_ * 4 * 2 + alignD_ * ESZ + 1024 + 12 * 256;
    int64_t budget = FA_UB_BUDGET - fixed;
    if (budget < 8192) { budget = 8192; }
    int64_t perKb = dDim_ * (8 + ESZ) + 44;   // qbr4 + kf4 + inQue ESZ + pbr32 + sc4 + part4 + mask4
    int64_t kb = budget / perKb;
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
    if (inQueElems < 256) { inQueElems = 256; }
    pipe_.InitBuffer(inQue_, 1, static_cast<uint32_t>(inQueElems * ESZ));
    pipe_.InitBuffer(outQue_, 1, static_cast<uint32_t>(alignD_ * ESZ + 256));
    pipe_.InitBuffer(qfBuf_, static_cast<uint32_t>(alignD_ * 4 + 256));
    pipe_.InitBuffer(qbrBuf_, static_cast<uint32_t>(kb_ * dDim_ * 4 + 256));
    pipe_.InitBuffer(kfBuf_, static_cast<uint32_t>(kb_ * dDim_ * 4 + 256));
    pipe_.InitBuffer(scoreBuf_, static_cast<uint32_t>(sPad_ * 4 + 256));
    int64_t redCap = kb_;
    if (sPad_ / FA_CHUNK + 8 > redCap) { redCap = sPad_ / FA_CHUNK + 8; }
    pipe_.InitBuffer(scBuf_, static_cast<uint32_t>(redCap * 4 + 256));
    pipe_.InitBuffer(partBuf_, static_cast<uint32_t>(redCap * 4 + 256));
    pipe_.InitBuffer(pbrBuf_, static_cast<uint32_t>(kb_ * 8 * 4 + 256));
    pipe_.InitBuffer(maskBuf_, static_cast<uint32_t>(kb_ * 4 + 256));
    pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(alignD_ * 4 + 256));
    pipe_.InitBuffer(smallBuf_, 1024);
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

// 构造广播后的 q（rows 行 × D），本行内所有 key 分块复用
template <typename T>
__aicore__ inline void FlashAttention<T>::BuildQbr(LocalTensor<float>& qf, LocalTensor<float>& qbr, int32_t rows)
{
    const int32_t nchunk = static_cast<int32_t>((dDim_ + FA_CHUNK - 1) / FA_CHUNK);
    const int32_t d8 = static_cast<int32_t>(dDim_ / 8);
    if (rows <= 0) { return; }
    for (int32_t c = 0; c < nchunk; ++c) {
        int32_t rem = static_cast<int32_t>(dDim_ - static_cast<int64_t>(c) * FA_CHUNK);
        if (rem > FA_CHUNK) { rem = FA_CHUNK; }
        UnaryRepeatParams rp;
        rp.dstBlkStride = 1;
        rp.srcBlkStride = 1;
        rp.dstRepStride = static_cast<uint16_t>(d8);
        rp.srcRepStride = 0;
        Adds(qbr[c * FA_CHUNK], qf[c * FA_CHUNK], 0.0f, rem, static_cast<uint8_t>(rows), rp);
    }
}

// 对 (cur, D) 的乘积张量沿 D 方向归约，得到 cur 个 score（紧凑写入 dst）
template <typename T>
__aicore__ inline void FlashAttention<T>::ReduceScores(int32_t cur, LocalTensor<float>& kf,
                                                       LocalTensor<float>& part, LocalTensor<float>& sc,
                                                       LocalTensor<float> dst)
{
    const int32_t nchunk = static_cast<int32_t>((dDim_ + FA_CHUNK - 1) / FA_CHUNK);
    const int32_t d8 = static_cast<int32_t>(dDim_ / 8);
    if (nchunk == 1) {
        // D <= 64：单次归约即可，结果直接写入整行 score
        WholeReduceSum(dst, kf, static_cast<int32_t>(dDim_), cur, 1, 1, d8);
        // reduce 写回需 drain 后才能被后续指令读取
        PipeBarrier<PIPE_V>();
        return;
    }
    for (int32_t c = 0; c < nchunk; ++c) {
        int32_t rem = static_cast<int32_t>(dDim_ - static_cast<int64_t>(c) * FA_CHUNK);
        if (rem > FA_CHUNK) { rem = FA_CHUNK; }
        // 每个 repeat 处理一个 key 行（D 的一段），结果紧凑写入 part[0..cur)
        WholeReduceSum(part, kf[c * FA_CHUNK], rem, cur, 1, 1, d8);
        // reduce 写回需 drain 后才能被后续指令读取
        PipeBarrier<PIPE_V>();
        if (c == 0) {
            Adds(sc, part, 0.0f, cur);
        } else {
            Add(sc, sc, part, cur);
        }
    }
    Adds(dst, sc, 0.0f, cur);
}

// PV 行广播点乘：vf[j, d] *= pbr[j]（pbr 为 Brcb 后的每行一个 32B 广播块）
template <typename T>
__aicore__ inline void FlashAttention<T>::RowMulBroadcast(LocalTensor<float>& vf, LocalTensor<float>& pbr, int32_t rows)
{
    const int32_t nchunk = static_cast<int32_t>((dDim_ + FA_CHUNK - 1) / FA_CHUNK);
    const int32_t d8 = static_cast<int32_t>(dDim_ / 8);
    for (int32_t c = 0; c < nchunk; ++c) {
        int32_t rem = static_cast<int32_t>(dDim_ - static_cast<int64_t>(c) * FA_CHUNK);
        if (rem > FA_CHUNK) { rem = FA_CHUNK; }
        BinaryRepeatParams rp;
        rp.dstBlkStride = 1;
        rp.src0BlkStride = 1;
        rp.src1BlkStride = 0;
        rp.dstRepStride = static_cast<uint8_t>(d8);
        rp.src0RepStride = static_cast<uint8_t>(d8);
        rp.src1RepStride = 1;
        Mul(vf[c * FA_CHUNK], vf[c * FA_CHUNK], pbr, rem, static_cast<uint8_t>(rows), rp);
        }
}

// 对 (rows, D) 的 work 张量沿行方向树形归约，得到 D 个列和并累加到 acc
template <typename T>
__aicore__ inline void FlashAttention<T>::ReduceRowsToAcc(LocalTensor<float>& work, int32_t rows, LocalTensor<float>& acc)
{
    const int32_t D = static_cast<int32_t>(dDim_);
    int32_t r = rows;
    while (r > 1) {
        int32_t half = r >> 1;
        // 同址树形归约：每步写回需 drain 后下一轮才能读取（参照官方 RowMax 模式）
        Add(work, work, work[half * D], half * D);
        PipeBarrier<PIPE_V>();
        if (r & 1) {
            Adds(work[half * D], work[(r - 1) * D], 0.0f, D);
            PipeBarrier<PIPE_V>();
        }
        r = half + (r & 1);
    }
    if (rows > 0) {
        Add(acc, acc, work, D);
        }
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
    LocalTensor<float> score = scoreBuf_.Get<float>();
    LocalTensor<float> sc = scBuf_.Get<float>();
    LocalTensor<float> part = partBuf_.Get<float>();
    LocalTensor<float> pbr = pbrBuf_.Get<float>();
    LocalTensor<float> maskf = maskBuf_.Get<float>();
    LocalTensor<float> acc = accBuf_.Get<float>();
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

    // 预填充整行 scores 为填充哨兵值（未写入的位置 = 0 概率），32B 对齐
    Duplicate(score, FA_PAD_NEG, static_cast<int32_t>(sPad_));

    // 构造广播后的 q（KB 行 × D），本行内复用
    BuildQbr(qf, qbr, static_cast<int32_t>(kb_));

    // ---- scores pass ----
    for (int64_t jb = 0; jb < sDim_; jb += kb_) {
        int32_t cur = static_cast<int32_t>((sDim_ - jb) < kb_ ? (sDim_ - jb) : kb_);
        // 因果：只计算 j <= i 的 key，其余保持为填充哨兵（等价 -inf）
        if (causal_) {
            int64_t valid = i - jb + 1;
            if (valid <= 0) { break; }
            if (valid < cur) { cur = static_cast<int32_t>(valid); }
        }
        CopyIn(kGm_, kBase + jb * dDim_, cur * static_cast<int32_t>(dDim_), kf);
        Mul(kf, kf, qbr, cur * static_cast<int32_t>(dDim_));
        ReduceScores(cur, kf, part, sc, score[jb]);

        // 加性掩码
        if (hasMask_) {
            CopyIn(mGm_, mBase + jb, cur, maskf);
            Add(score[jb], score[jb], maskf, cur);
        }
    }

    // TODO: 实现 Compute 逻辑（行内稳定 softmax：max / exp / sum）
    // ---- 行内稳定 softmax（分块 reduce + 向量累加，机制与官方实现一致）----
    float m = 0.0f;
    float l = 1.0f;
    {
    const int32_t nchunkS = static_cast<int32_t>(sPad_ / FA_CHUNK);
    Duplicate(mv, FA_TAIL_NEG, 8);
    for (int32_t c = 0; c < nchunkS; ++c) {
        WholeReduceMax(part, score[c * FA_CHUNK], FA_CHUNK, 1, 1, 1, 1);
        Max(mv, mv, part, 1);
    }
    PipeBarrier<PIPE_ALL>();
    m = mv.GetValue(0);
    // 全屏蔽行防御：行最大非有限时按 0 处理，避免 NaN（题目 3.7）
    if (m != m || m <= FA_TAIL_NEG) {
        m = 0.0f;
    }
    Adds(score, score, -m, static_cast<int32_t>(sPad_));
    Exp(score, score, static_cast<int32_t>(sPad_));

    Duplicate(sv, 0.0f, 8);
    for (int32_t c = 0; c < nchunkS; ++c) {
        WholeReduceSum(part, score[c * FA_CHUNK], FA_CHUNK, 1, 1, 1, 1);
        Add(sv, sv, part, 1);
    }
    PipeBarrier<PIPE_ALL>();
    l = sv.GetValue(0);
    // 分母防御：非有限或非正时按 1 处理（全屏蔽行输出 0 而非 NaN）
    if (l != l || l <= 0.0f) {
        l = 1.0f;
    }
    }

    // ---- PV: out = (sum_j p_j * V_j) / l ----
    Duplicate(acc, 0.0f, static_cast<int32_t>(alignD_));
    for (int64_t jb = 0; jb < sDim_; jb += kb_) {
        int32_t cur = static_cast<int32_t>((sDim_ - jb) < kb_ ? (sDim_ - jb) : kb_);
        if (causal_) {
            int64_t valid = i - jb + 1;
            if (valid <= 0) { break; }
            if (valid < cur) { cur = static_cast<int32_t>(valid); }
        }
        CopyIn(vGm_, kBase + jb * dDim_, cur * static_cast<int32_t>(dDim_), kf);
        // p -> 每行一个 32B 广播块
        int32_t nrep = (cur + 7) / 8;
        Brcb(pbr, score[jb], static_cast<uint8_t>(nrep), {1, 8});
        RowMulBroadcast(kf, pbr, cur);
        ReduceRowsToAcc(kf, cur, acc);
    }
    const float invl = 1.0f / l;
    PipeBarrier<PIPE_ALL>();
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