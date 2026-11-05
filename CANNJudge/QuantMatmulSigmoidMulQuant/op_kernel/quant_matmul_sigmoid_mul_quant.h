/*!
 * \file quant_matmul_sigmoid_mul_quant.h
 * \brief QuantMatmulSigmoidMulQuant 算子 kernel 类定义
 */

#ifndef QUANTMATMULSIGMOIDMULQUANT_H
#define QUANTMATMULSIGMOIDMULQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_data.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_key.h"

namespace NsQuantMatmulSigmoidMulQuant {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t QMSQ_KBLK = 32;
constexpr int64_t QMSQ_MAXM = 32;
constexpr int64_t QMSQ_MAXN = 256;
constexpr int64_t QMSQ_REDUCE_MAX = 64;

constexpr uint64_t QMSQ_M_MASK = 0xFFFFFFULL;
constexpr uint64_t QMSQ_K_MASK = 0xFFFFFFULL;
constexpr uint64_t QMSQ_N_MASK = 0xFFFFFFULL;
constexpr uint64_t QMSQ_CM_MASK = 0xFFFULL;
constexpr uint64_t QMSQ_CN_MASK = 0xFFFULL;

template <typename T>
class QuantMatmulSigmoidMulQuant {
public:
    __aicore__ inline QuantMatmulSigmoidMulQuant(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR bias, GM_ADDR y, GM_ADDR yScale, const QuantMatmulSigmoidMulQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTiling(const QuantMatmulSigmoidMulQuantTilingData* td);
    __aicore__ inline void SetupCore(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale);
    __aicore__ inline void AllocBuffers();

    __aicore__ inline void LoadX1Block(int64_t mBase, int64_t kBase, int64_t mCnt, int64_t kCnt);
    __aicore__ inline void LoadX2Block(int64_t kBase, int64_t nBase, int64_t kCnt, int64_t nCnt);
    __aicore__ inline void MatmulRows(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt);
    __aicore__ inline void BuildU(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt);
    __aicore__ inline float LocalAbsMax(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt);
    __aicore__ inline void PostChunk(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt, float yScale);

    TPipe pipe;
    TBuf<TPosition::VECCALC> x1Buf_;
    TBuf<TPosition::VECCALC> x1hBuf_;
    TBuf<TPosition::VECCALC> x1fBuf_;
    TBuf<TPosition::VECCALC> x2Buf_;
    TBuf<TPosition::VECCALC> x2hBuf_;
    TBuf<TPosition::VECCALC> x2fBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> t1Buf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    TBuf<TPosition::VECCALC> t3Buf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> y8Buf_;
    TBuf<TPosition::VECCALC> redBuf_;
    TBuf<TPosition::VECCALC> statBuf_;

    GlobalTensor<T> x1GM_;
    GlobalTensor<T> x2GM_;
    GlobalTensor<float> x1ScaleGM_;
    GlobalTensor<float> x2ScaleGM_;
    GlobalTensor<int8_t> yGM_;
    GlobalTensor<float> yScaleGM_;
    // 跨核 abs_max 归约区：float 视图覆盖 y GM 前 8*blockNum 个 float 位
    GlobalTensor<float> redGM_;

    int64_t M_ = 1, K_ = 1, N_ = 1;
    int64_t chunkM_ = 1, chunkN_ = 1;
    int64_t totalNum_ = 1;
    int64_t blockNum_ = 1, blockIdx_ = 0;
    bool transposeX1_ = false, transposeX2_ = false;
    float merged_ = 1.0f;
    int64_t kBlk_ = QMSQ_KBLK;
};

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::ParseTiling(const QuantMatmulSigmoidMulQuantTilingData* td)
{
    // TODO: 实现 ParseTiling 逻辑
    uint64_t blockPacked = static_cast<uint64_t>(td->blockFactor);
    uint64_t ubPacked = static_cast<uint64_t>(td->ubFactor);

    M_ = static_cast<int64_t>(blockPacked & QMSQ_M_MASK);
    K_ = static_cast<int64_t>((blockPacked >> 24) & QMSQ_K_MASK);
    transposeX1_ = ((blockPacked >> 48) & 0x1ULL) != 0;
    transposeX2_ = ((blockPacked >> 49) & 0x1ULL) != 0;

    N_ = static_cast<int64_t>(ubPacked & QMSQ_N_MASK);
    chunkM_ = static_cast<int64_t>((ubPacked >> 24) & QMSQ_CM_MASK);
    chunkN_ = static_cast<int64_t>((ubPacked >> 36) & QMSQ_CN_MASK);

    if (M_ <= 0) { M_ = 1; }
    if (K_ <= 0) { K_ = 1; }
    if (N_ <= 0) { N_ = 1; }
    if (chunkM_ <= 0) { chunkM_ = 1; }
    if (chunkN_ <= 0) { chunkN_ = 1; }
    if (chunkM_ > QMSQ_MAXM) { chunkM_ = QMSQ_MAXM; }
    if (chunkN_ > QMSQ_MAXN) { chunkN_ = QMSQ_MAXN; }

    totalNum_ = M_ * N_;
    if (totalNum_ <= 0) { totalNum_ = 1; }

    if (kBlk_ > K_) { kBlk_ = K_; }
    if (kBlk_ < 1) { kBlk_ = 1; }
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::SetupCore(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale)
{
    // TODO: 实现 SetupCore 逻辑
    x1GM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1), M_ * K_);
    x2GM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2), K_ * N_);
    x1ScaleGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Scale), 1);
    x2ScaleGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Scale), 1);
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y), totalNum_);
    yScaleGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yScale), 1);
    // 跨核 abs_max 归约区：float 视图复用 y GM 前 blockNum_*8 个 int8 位
    // （1 个 float = 8 个 int8 元素）。host 已保证 totalNum >= blockNum*8，
    // 故该区间与 y 输出区的重叠部分在第二道 SyncAll 之后才被覆写。
    redGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), blockNum_ * 8);
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::AllocBuffers()
{
    // TODO: 实现 AllocBuffers 逻辑
    int64_t mCnt = chunkM_;
    int64_t nCnt = chunkN_;
    if (mCnt < 1) { mCnt = 1; }
    if (nCnt < 1) { nCnt = 1; }

    const uint32_t MIN_BUF = 512;

    auto alignUp32 = [](uint32_t v) -> uint32_t {
        return (v + 31U) & ~31U;
    };
    // 向量指令 count 须 8 元素对齐
    auto align8 = [](int64_t v) -> int64_t {
        return (v + 7) & ~7L;
    };

    int64_t mAl = align8(mCnt);
    int64_t nAl = align8(nCnt);
    int64_t kAl = align8(kBlk_);
    // 计算链行距：32 的倍数（满足 Cast/DataCopyPad 的 32B 对齐硬约束）
    int64_t stride = (nAl + 31) & ~31L;

    // 计算块统一按最大元素数 mAl*stride 的 float 尺寸分配，
    // 兼容 float/half/int32/int8 多类型复用（避免尺寸不足或未对齐）
    uint32_t calcBytes = alignUp32(static_cast<uint32_t>(mAl * stride * static_cast<int64_t>(sizeof(float))));
    if (calcBytes < MIN_BUF) { calcBytes = MIN_BUF; }

    // 输入中转块：x1=[mCnt][kAl]，x2=[kAl][stride]，按 float 尺寸分配以兼容
    // int8/half/float/int32。x2 行距取 stride（32 倍数）而非 nAl：尾块
    // nAl<32 时 stride=32，若按 nAl 分配则 Cast<float> 视图会越界。
    uint32_t x1Bytes = alignUp32(static_cast<uint32_t>(mAl * kAl * static_cast<int64_t>(sizeof(float))));
    uint32_t x2Bytes = alignUp32(static_cast<uint32_t>(kAl * stride * static_cast<int64_t>(sizeof(float))));
    if (x1Bytes < MIN_BUF) { x1Bytes = MIN_BUF; }
    if (x2Bytes < MIN_BUF) { x2Bytes = MIN_BUF; }

    // half 中转块容量必须同时覆盖两种用途，否则越界写踩相邻 UB：
    //   1) MatmulRows 的 int8->half Cast：x1 需 mAl*kAl、x2 需 kAl*stride 个 half
    //   2) BuildU 复用作 fp16 中间 buffer（h16/g16）：需 mAl*stride 个 half
    // x1fBuf_/x2fBuf_ 与 half 视图分离（原实现共用一个 TBuf，float 视图写会覆盖
    // half 视图尚未读走的低位，属读写重叠）
    uint32_t x1hBytes = alignUp32(static_cast<uint32_t>(
        2 * ((mAl * kAl > mAl * stride) ? (mAl * kAl) : (mAl * stride))));
    uint32_t x2hBytes = alignUp32(static_cast<uint32_t>(
        2 * ((kAl * stride > mAl * stride) ? (kAl * stride) : (mAl * stride))));
    if (x1hBytes < MIN_BUF) { x1hBytes = MIN_BUF; }
    if (x2hBytes < MIN_BUF) { x2hBytes = MIN_BUF; }

    // int8 输出中转块：行距与计算链一致取 stride（32 元素倍数），
    // 否则 Cast 与 DataCopyPad 的 src 非 32B 对齐会 abort。
    // M 尾块 mCnt 最大为 chunkM_，故容量按 mAl 行预留。
    uint32_t y8Bytes = alignUp32(static_cast<uint32_t>(mAl * stride));
    if (y8Bytes < MIN_BUF) { y8Bytes = MIN_BUF; }

    pipe.InitBuffer(x1Buf_, x1Bytes);
    pipe.InitBuffer(x1hBuf_, x1hBytes);
    pipe.InitBuffer(x1fBuf_, x1Bytes);
    pipe.InitBuffer(x2Buf_, x2Bytes);
    pipe.InitBuffer(x2hBuf_, x2hBytes);
    pipe.InitBuffer(x2fBuf_, x2Bytes);
    pipe.InitBuffer(accBuf_, calcBytes);
    pipe.InitBuffer(t1Buf_, calcBytes);
    pipe.InitBuffer(t2Buf_, calcBytes);
    pipe.InitBuffer(t3Buf_, calcBytes);
    pipe.InitBuffer(yBuf_, calcBytes);
    pipe.InitBuffer(y8Buf_, y8Bytes);
    pipe.InitBuffer(redBuf_, MIN_BUF);
    pipe.InitBuffer(statBuf_, MIN_BUF);
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR bias, GM_ADDR y, GM_ADDR yScale, const QuantMatmulSigmoidMulQuantTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    (void)bias;
    ParseTiling(tilingData);

    blockNum_ = static_cast<int64_t>(GetBlockNum());
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    if (blockNum_ < 1) { blockNum_ = 1; }
    if (blockIdx_ < 0) { blockIdx_ = 0; }
    // 归约区尺寸不能超过 y GM 的 float 容量（8 个 int8 元素承载 1 个 float）
    if (totalNum_ < blockNum_ * 8) { blockNum_ = 1; blockIdx_ = 0; }

    SetupCore(x1, x2, x1Scale, x2Scale, y, yScale);
    AllocBuffers();

    float s1 = x1ScaleGM_.GetValue(0);
    float s2 = x2ScaleGM_.GetValue(0);
    merged_ = s1 * s2;
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::Process()
{
    // TODO: 实现 Process 逻辑
    // 多核按 M 块轮询切分（mChunk = blockIdx, blockIdx+blockNum, ...）：
    // 每核负责全部 N 列，x1Scale/x2Scale 每核只读一次。
    // per-tensor 量化的全局 abs_max 须跨核归约：无独立 workspace 通道，
    // 复用 y GM 前 blockNum 个 float 位作每核 local abs_max 暂存区
    // （8 个 int8 元素 = 1 个 float，host 已保证 totalNum >= blockNum*8），
    // 双 SyncAll 完成「写 local → 读全部求 max」握手。
    // 第二道 SyncAll 保证所有核都读完归约区后，才开始写 y 输出
    // （否则后完成的核可能覆写归约区而其它核尚未读取）。
    float myMax = 0.0f;
    for (int64_t m = blockIdx_ * chunkM_; m < M_; m += blockNum_ * chunkM_) {
        int64_t mCnt = M_ - m;
        if (mCnt > chunkM_) { mCnt = chunkM_; }
        for (int64_t n = 0; n < N_; n += chunkN_) {
            int64_t nCnt = N_ - n;
            if (nCnt > chunkN_) { nCnt = chunkN_; }
            float cm = LocalAbsMax(m, mCnt, n, nCnt);
            if (cm > myMax) { myMax = cm; }
        }
    }

    // 写 local abs_max 到归约区本核槽位（仅多核需要；单核时 myMax 即全局值，
    // 且 y GM 可能不足 blockNum*8 字节，写入会越界）
    if (blockNum_ > 1) {
        LocalTensor<float> lb = statBuf_.Get<float>();
        lb.SetValue(0, myMax);
        event_t evS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evS);
        WaitFlag<HardEvent::S_MTE3>(evS);
        DataCopyExtParams rp{1, 4, 0, 0, 0};
        DataCopyPad(redGM_[blockIdx_ * 8], lb, rp);
        event_t evM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(evM);
        WaitFlag<HardEvent::MTE3_MTE2>(evM);
    }

    if (blockNum_ > 1) {
        SyncAll();
    }

    // 读回全部核的 local abs_max 求全局最大值（本核槽位已由 myMax 覆盖）
    float gMax = myMax;
    if (blockNum_ > 1) {
        for (int64_t b = 0; b < blockNum_; ++b) {
            if (b == blockIdx_) { continue; }
            float v = redGM_.GetValue(static_cast<uint32_t>(b * 8));
            if (v > gMax) { gMax = v; }
        }
    }

    float yScale = gMax / 127.0f;
    if (!(yScale > 0.0f)) { yScale = 1.0f; }

    // 仅 block 0 写 yScale 输出，避免多核重复写
    if (blockIdx_ == 0) {
        LocalTensor<float> ysBuf = statBuf_.Get<float>();
        ysBuf.SetValue(0, yScale);
        event_t evS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evS);
        WaitFlag<HardEvent::S_MTE3>(evS);
        DataCopyExtParams yp{1, 4, 0, 0, 0};
        DataCopyPad(yScaleGM_[0], ysBuf, yp);
        event_t evM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(evM);
        WaitFlag<HardEvent::MTE3_MTE2>(evM);
    }

    // 第二道 SyncAll：确保所有核都读完归约区后才开始覆写 y 输出区
    if (blockNum_ > 1) {
        SyncAll();
    }

    for (int64_t m = blockIdx_ * chunkM_; m < M_; m += blockNum_ * chunkM_) {
        int64_t mCnt = M_ - m;
        if (mCnt > chunkM_) { mCnt = chunkM_; }
        for (int64_t n = 0; n < N_; n += chunkN_) {
            int64_t nCnt = N_ - n;
            if (nCnt > chunkN_) { nCnt = chunkN_; }
            PostChunk(m, mCnt, n, nCnt, yScale);
        }
    }

    event_t evEnd = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(evEnd);
    WaitFlag<HardEvent::MTE3_MTE2>(evEnd);
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::LoadX1Block(int64_t mBase, int64_t kBase, int64_t mCnt, int64_t kCnt)
{
    // TODO: 实现 LoadX1Block 逻辑
    // transposeX1_=false：x1 物理 (M,K)，UB 布局 [mCnt][kAl]（行主序）
    // transposeX1_=true ：x1 物理 (K,M)，UB 布局 [kAl][mAl]（按 k 行、m 列）
    LocalTensor<T> dst = x1Buf_.Get<T>();
    int64_t kAl = (kCnt + 7) & ~7L;
    int64_t mAl = (mCnt + 7) & ~7L;
    if (!transposeX1_) {
        for (int64_t m = 0; m < mCnt; ++m) {
            for (int64_t k = 0; k < kAl; ++k) {
                int64_t gmIdx = (k < kCnt) ? ((mBase + m) * K_ + kBase + k) : -1;
                dst.SetValue(static_cast<uint32_t>(m * kAl + k),
                             (gmIdx >= 0) ? x1GM_.GetValue(static_cast<uint32_t>(gmIdx)) : static_cast<T>(0));
            }
        }
    } else {
        for (int64_t k = 0; k < kAl; ++k) {
            for (int64_t m = 0; m < mAl; ++m) {
                int64_t gmIdx = (k < kCnt && m < mCnt) ? ((kBase + k) * M_ + mBase + m) : -1;
                dst.SetValue(static_cast<uint32_t>(k * mAl + m),
                             (gmIdx >= 0) ? x1GM_.GetValue(static_cast<uint32_t>(gmIdx)) : static_cast<T>(0));
            }
        }
    }
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::LoadX2Block(int64_t kBase, int64_t nBase, int64_t kCnt, int64_t nCnt)
{
    // TODO: 实现 LoadX2Block 逻辑
    // 统一整理为 UB 布局 [kAl][stride]（行主序，行距 = stride，32 元素倍数）。
    // 行距必须与 MatmulRows 的 Axpy 读取偏移一致：dav_c220 的 Cast 与
    // DataCopyPad 强制 32B 对齐，故 stride 取 32 倍数而非仅 8 元素对齐。
    LocalTensor<T> dst = x2Buf_.Get<T>();
    int64_t kAl = (kCnt + 7) & ~7L;
    int64_t nAl = (nCnt + 7) & ~7L;
    int64_t stride = (nAl + 31) & ~31L;
    for (int64_t k = 0; k < kAl; ++k) {
        for (int64_t n = 0; n < stride; ++n) {
            int64_t gmIdx = (k < kCnt && n < nCnt)
                ? (transposeX2_ ? (nBase + n) * K_ + (kBase + k) : (kBase + k) * N_ + (nBase + n))
                : -1;
            dst.SetValue(static_cast<uint32_t>(k * stride + n),
                         (gmIdx >= 0) ? x2GM_.GetValue(static_cast<uint32_t>(gmIdx)) : static_cast<T>(0));
        }
    }
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::MatmulRows(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt)
{
    // TODO: 实现 MatmulRows 逻辑
    LocalTensor<float> acc = accBuf_.Get<float>();

    LocalTensor<T> x1Raw = x1Buf_.Get<T>();
    LocalTensor<T> x2Raw = x2Buf_.Get<T>();
    LocalTensor<half> x1h = x1hBuf_.Get<half>();
    LocalTensor<half> x2h = x2hBuf_.Get<half>();
    LocalTensor<float> x1f = x1fBuf_.Get<float>();
    LocalTensor<float> x2f = x2fBuf_.Get<float>();

    // 行距统一取 32 的倍数（而非仅 8 元素对齐）：PostChunk 需对每行做
    // Cast<int8_t> 写 GM，dav_c220 的 Cast 与 DataCopyPad 均强制 src/dst
    // 32B 对齐，nAl 非 32 倍数时行首地址不对齐会 abort。
    int32_t nAl = static_cast<int32_t>((nCnt + 7) & ~7);
    int64_t stride = (nAl + 31) & ~31L;

    // acc 按对齐宽度清零，避免未初始化内存参与归约
    Duplicate(acc, 0.0f, static_cast<int32_t>(mCnt * stride));

    for (int64_t kBase = 0; kBase < K_; kBase += kBlk_) {
        int64_t kCnt = K_ - kBase;
        if (kCnt > kBlk_) { kCnt = kBlk_; }
        int32_t kAl = static_cast<int32_t>((kCnt + 7) & ~7);

        // LoadX*Block 已按 [.][kAl/nAl] 对齐布局写入，填充区为 0
        LoadX1Block(mBase, kBase, mCnt, kCnt);
        LoadX2Block(kBase, nBase, kCnt, nCnt);

        // x1 两种布局的 Cast 元素数：非转置 [mCnt][kAl]，转置 [kAl][mAl]
        int64_t mAl = (mCnt + 7) & ~7L;
        int32_t x1CastCnt = transposeX1_ ? static_cast<int32_t>(kAl * mAl)
                                         : static_cast<int32_t>(mCnt * kAl);
        // int8 输入经 int8->half->float 两段 Cast；half 输入（schMode=0）直接进 float
        if constexpr (sizeof(T) == 1) {
            Cast(x1h, x1Raw, RoundMode::CAST_NONE, x1CastCnt);
            Cast(x2h, x2Raw, RoundMode::CAST_NONE, static_cast<int32_t>(kAl * stride));
            Cast(x2f, x2h, RoundMode::CAST_NONE, static_cast<int32_t>(kAl * stride));
            Cast(x1f, x1h, RoundMode::CAST_NONE, x1CastCnt);
        } else {
            Cast(x2f, x2Raw, RoundMode::CAST_NONE, static_cast<int32_t>(kAl * stride));
            Cast(x1f, x1Raw, RoundMode::CAST_NONE, x1CastCnt);
        }

        for (int64_t m = 0; m < mCnt; ++m) {
            for (int64_t k = 0; k < kCnt; ++k) {
                int64_t aIdx = transposeX1_ ? (k * mAl + m) : (m * kAl + k);
                float av = x1f.GetValue(static_cast<uint32_t>(aIdx));
                Axpy<float, float>(acc[static_cast<uint32_t>(m * stride)], x2f[static_cast<uint32_t>(k * stride)], av, nAl);
            }
        }
    }
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::BuildU(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt)
{
    // TODO: 实现 BuildU 逻辑
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> h = t1Buf_.Get<float>();
    LocalTensor<float> z = t2Buf_.Get<float>();
    LocalTensor<float> e = t3Buf_.Get<float>();
    LocalTensor<float> u = yBuf_.Get<float>();
    LocalTensor<half> h16 = x1hBuf_.Get<half>();
    LocalTensor<half> g16 = x2hBuf_.Get<half>();

    // count 须与 MatmulRows 的 stride 一致：行距取 32 倍数，再按整块补齐
    int64_t nAl = (nCnt + 7) & ~7L;
    int64_t stride = (nAl + 31) & ~31L;
    int64_t cntA = mCnt * stride;
    int32_t cntAl = static_cast<int32_t>((cntA + 31) & ~31L);

    Muls(h, acc, merged_, cntAl);
    Cast(h16, h, RoundMode::CAST_NONE, cntAl);
    Cast(h, h16, RoundMode::CAST_NONE, cntAl);

    // sigmoid：z = -h 时 clamp exp 参数到 [-88, 88] 防溢出，h 本身不 clamp
    //（golden 的 u = sigmoid(h)*h，h 大时 u 随 h 线性增长，clamp h 会截断该行为）
    Muls(z, h, -1.0f, cntAl);
    Mins(z, z, 88.0f, cntAl);
    Maxs(z, z, -88.0f, cntAl);
    Exp(e, z, cntAl);
    Adds(e, e, 1.0f, cntAl);
    Duplicate(acc, 1.0f, cntAl);
    Div(z, acc, e, cntAl);
    Cast(g16, z, RoundMode::CAST_NONE, cntAl);
    Cast(z, g16, RoundMode::CAST_NONE, cntAl);

    Mul(u, z, h, cntAl);
    Cast(h16, u, RoundMode::CAST_NONE, cntAl);
    Cast(u, h16, RoundMode::CAST_NONE, cntAl);
}

template <typename T>
__aicore__ inline float QuantMatmulSigmoidMulQuant<T>::LocalAbsMax(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt)
{
    // TODO: 实现 LocalAbsMax 逻辑
    MatmulRows(mBase, mCnt, nBase, nCnt);
    BuildU(mBase, mCnt, nBase, nCnt);

    LocalTensor<float> u = yBuf_.Get<float>();
    LocalTensor<float> rmax = redBuf_.Get<float>();
    LocalTensor<float> fold = t2Buf_.Get<float>();

    // count 须与 MatmulRows/BuildU 的 stride 一致（32 倍数行距 + 整块补齐）
    int64_t nAl = ((nCnt + 7) & ~7L);
    int64_t stride = (nAl + 31) & ~31L;
    int64_t cntA = mCnt * stride;
    int32_t cntAl = static_cast<int32_t>((cntA + 31) & ~31L);
    Abs(u, u, static_cast<int32_t>(cntA));
    Duplicate(fold, 0.0f, cntAl);
    Max(fold, fold, u, static_cast<int32_t>(cntA));
    // 归约折叠（对齐 MultiAddRmsNormDynamicQuant::ReduceMaxScalar 真机惯用法）：
    // 固定 REP=64 宽的 repeat 折叠 + 尾部并入 + WholeReduceMax 收尾。
    // 不可用树形折半：cntAl=align32(cntA) 非 2 的幂时折半链会落到非 8 倍数
    // （如 640→20）或提前停在 <32，导致 Max count 非法 / 越界读未初始化 UB
    // （曾致 yScale=0xEEEEEEEE=-3.697e28 垃圾值）。
    constexpr int32_t REP = 64;
    if (cntAl > REP) {
        const int32_t k = cntAl / REP;
        const int32_t rems = cntAl - k * REP;
        if (k > 1) {
            Max(fold, fold[static_cast<uint32_t>(REP)], fold, REP, k - 1, {1, 1, 1, 0, 8, 0});
        }
        if (rems > 0) {
            Max(fold, fold[static_cast<uint32_t>(k * REP)], fold, rems);
        }
        WholeReduceMax(rmax, fold, REP, 1, 8, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        WholeReduceMax(rmax, fold, cntAl, 1, 8, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    }
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(ev);
    WaitFlag<HardEvent::V_S>(ev);
    return rmax.GetValue(0);
}

template <typename T>
__aicore__ inline void QuantMatmulSigmoidMulQuant<T>::PostChunk(int64_t mBase, int64_t mCnt, int64_t nBase, int64_t nCnt, float yScale)
{
    // TODO: 实现 PostChunk 逻辑
    MatmulRows(mBase, mCnt, nBase, nCnt);
    BuildU(mBase, mCnt, nBase, nCnt);

    LocalTensor<float> u = yBuf_.Get<float>();
    LocalTensor<half> q16 = x2hBuf_.Get<half>();
    LocalTensor<int32_t> s32 = t3Buf_.Get<int32_t>();
    LocalTensor<float> f = t1Buf_.Get<float>();

    // count 与 BuildU 一致：行距 32 倍数 + 整块补齐
    int64_t nAl = (nCnt + 7) & ~7L;
    int64_t stride = (nAl + 31) & ~31L;
    int64_t cntA = mCnt * stride;
    int32_t cntAl = static_cast<int32_t>((cntA + 31) & ~31L);
    // 必须用真除法（而非倒数乘法）对齐 golden：gen_data.py 的 u32 / y_scale。
    // 倒数乘法在 tie 附近会差 1 ULP（实测 63.4999962 -> 63.5），
    // 使 CAST_RINT 多进一位，产出 64 vs golden 63。
    // inv 借用 accBuf_（calcBytes 容量足够；BuildU 之后 acc 已无用，且与 yBuf_ 无重叠）
    LocalTensor<float> inv = accBuf_.Get<float>();
    Duplicate(inv, yScale, cntAl);
    Div(u, u, inv, cntAl);
    Mins(u, u, 127.0f, cntAl);
    Maxs(u, u, -128.0f, cntAl);
    Cast(s32, u, RoundMode::CAST_RINT, cntAl);
    Cast(f, s32, RoundMode::CAST_NONE, cntAl);
    Cast(q16, f, RoundMode::CAST_NONE, cntAl);

    // y8 落到 32B 对齐行距的专用中转块（见 AllocBuffers 的 y8Bytes 说明）：
    // Cast 与 DataCopyPad 均强制 src/dst 32B 对齐，故 q16 与 y8 的行距都取
    // 与 MatmulRows 一致的 stride（32 元素倍数）。
    const int64_t y8Stride = stride;
    LocalTensor<int8_t> y8 = y8Buf_.Get<int8_t>();
    for (int64_t m = 0; m < mCnt; ++m) {
        Cast(y8[static_cast<uint32_t>(m * y8Stride)],
             q16[static_cast<uint32_t>(m * stride)], RoundMode::CAST_RINT, static_cast<int32_t>(stride));
    }

    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);

    // 多核写 GM：每核负责的 M 行段在 y GM 内互不重叠，故用 MTE3 DataCopyPad
    // 逐行一维 burst 写出（blockCount=1, blockLen=nCnt 字节）。
    // 不可用多核逐元素 SetValue 直写（BatchNorm3D/QBMv3 两次确证丢失）。
    // 不用二维 burst：写 GM 的 srcStride 单位为 32B 块（见
    // kernel_check_data_copy_overflow.h 的 * DEFAULT_C0_SIZE），int8 行距
    // 无法任意表达；一维逐行与 11 个已过评测算子的惯用法一致。
    for (int64_t m = 0; m < mCnt; ++m) {
        DataCopyExtParams yp{1, static_cast<uint32_t>(nCnt), 0, 0, 0};
        DataCopyPad(yGM_[(mBase + m) * N_ + nBase], y8[static_cast<uint32_t>(m * y8Stride)], yp);
    }
    event_t evM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(evM);
    WaitFlag<HardEvent::MTE3_MTE2>(evM);
}

} // namespace NsQuantMatmulSigmoidMulQuant
#endif
