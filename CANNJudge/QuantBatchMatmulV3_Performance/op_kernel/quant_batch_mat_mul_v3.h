/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 算子 kernel 类定义
 */

#ifndef QUANTBATCHMATMULV3_H
#define QUANTBATCHMATMULV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_batch_mat_mul_v3_tiling_data.h"
#include "quant_batch_mat_mul_v3_tiling_key.h"

namespace NsQuantBatchMatMulV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t QBMM_MAXM = 16;
constexpr int64_t QBMM_MAXN = 192;
// K 维分块大小：取 64 以摊薄 GM 搬运轮次（K=32768 时 512 轮 vs 32 分块的 1024 轮）。
// UB 核算（mAl=16, stride=192, kAl=64）：
//   x1=16*64*4=4KB, x1h=16*64*2=2KB, x2=64*192*4=48KB, x2h=64*192*2=24KB,
//   acc=16*192*4=12KB, out=16*192*2=6KB, 其余约 2KB → 合计约 98KB < 192KB。
constexpr int64_t QBMM_KBLK = 64;
constexpr uint32_t MIN_BUF = 512;

// TilingData 位域打包约定（与 op_host/quant_batch_mat_mul_v3_tiling.cpp 严格一致）：
//   totalNum    : [0:32)   = K（累加维度）
//                 [32:64)  = M（x1 行数 / out 行数）
//   blockFactor : [0:32)   = mCnt（每核每次处理的 M 行数 = chunkM）
//                 [32:64)  = bf（每核负责的 x1 元素数 = mCnt * K）
//   ubFactor    : [0:32)   = chunkN（每次 UB 循环处理的 N 列数）
//                 [32:64)  = N（x2 列数 / out 列数）
constexpr uint64_t DIM_MASK = 0xFFFFFFFFULL;

template <typename T>
class QuantBatchMatMulV3 {
public:
    __aicore__ inline QuantBatchMatMulV3(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, const QuantBatchMatMulV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTiling(const QuantBatchMatMulV3TilingData* td);
    __aicore__ inline void SetupGlobal(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out);
    __aicore__ inline void AllocBuffers();
    __aicore__ inline void LoadX1Block(int64_t mBase, int64_t kBase, int64_t mCnt, int64_t kCnt, int64_t kAl);
    __aicore__ inline void LoadX2Block(int64_t kBase, int64_t nBase, int64_t kCnt, int64_t nCnt, int64_t stride);
    __aicore__ inline void LoadScaleBlock(int64_t nBase, int64_t nCnt, int64_t stride);
    __aicore__ inline void LoadPertokenBlock(int64_t mBase, int64_t mCnt);
    __aicore__ inline void MatmulBlock(int64_t mCnt, int64_t kCnt, int64_t kAl, int64_t stride);
    __aicore__ inline void DequantBlock(int64_t mCnt, int64_t nCnt, int64_t stride);
    __aicore__ inline void WriteOutBlock(int64_t mBase, int64_t nBase, int64_t mCnt, int64_t nCnt, int64_t stride);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> x1Buf;
    TBuf<TPosition::VECCALC> x1hBuf;
    TBuf<TPosition::VECCALC> x1fBuf;
    TBuf<TPosition::VECCALC> x2Buf;
    TBuf<TPosition::VECCALC> x2hBuf;
    TBuf<TPosition::VECCALC> x2fBuf;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> scaleBuf;
    TBuf<TPosition::VECCALC> pertokenBuf;
    TBuf<TPosition::VECCALC> outBuf;

    GlobalTensor<int8_t> inputGMX;
    GlobalTensor<int8_t> inputGMX2;
    GlobalTensor<float> scaleGM;
    GlobalTensor<float> pertokenScaleGM;
    GlobalTensor<bfloat16_t> outGM;

    int64_t M_ = 1;
    int64_t K_ = 1;
    int64_t N_ = 1;
    int64_t mCnt_ = 1;
    int64_t chunkN_ = 1;
    int64_t kBlk_ = QBMM_KBLK;
    int64_t blockNum_ = 1;
    int64_t blockIdx_ = 0;
};

// ======================================================================
// ParseTiling：从 tilingData 解码位域参数，带保护性兜底
// ======================================================================
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::ParseTiling(const QuantBatchMatMulV3TilingData* td)
{
    uint64_t totalRaw = static_cast<uint64_t>(static_cast<int64_t>(td->totalNum));
    M_ = static_cast<int64_t>((totalRaw >> 32) & DIM_MASK);
    K_ = static_cast<int64_t>(totalRaw & DIM_MASK);

    uint64_t blockRaw = static_cast<uint64_t>(static_cast<int64_t>(td->blockFactor));
    mCnt_ = static_cast<int64_t>(blockRaw & DIM_MASK);

    uint64_t ubRaw = static_cast<uint64_t>(static_cast<int64_t>(td->ubFactor));
    N_ = static_cast<int64_t>((ubRaw >> 32) & DIM_MASK);
    chunkN_ = static_cast<int64_t>(ubRaw & DIM_MASK);

    if (M_ <= 0) { M_ = 1; }
    if (K_ <= 0) { K_ = 1; }
    if (N_ <= 0) { N_ = 1; }
    if (mCnt_ <= 0) { mCnt_ = 1; }
    if (mCnt_ > QBMM_MAXM) { mCnt_ = QBMM_MAXM; }
    if (chunkN_ <= 0) { chunkN_ = 1; }
    if (chunkN_ > QBMM_MAXN) { chunkN_ = QBMM_MAXN; }

    kBlk_ = (K_ < QBMM_KBLK) ? K_ : QBMM_KBLK;
    if (kBlk_ < 1) { kBlk_ = 1; }
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::SetupGlobal(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out)
{
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), M_ * K_);
    inputGMX2.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), K_ * N_);
    scaleGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), N_);
    pertokenScaleGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), M_);
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), M_ * N_);
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::AllocBuffers()
{
    auto alignUp32 = [](uint32_t v) -> uint32_t { return (v + 31U) & ~31U; };
    auto align8 = [](int64_t v) -> int64_t { return (v + 7) & ~7L; };
    auto align32 = [](int64_t v) -> int64_t { return (v + 31) & ~31L; };

    int64_t mAl = align8(mCnt_);
    int64_t kAl = align32(kBlk_);
    int64_t nAl = align8(chunkN_);
    // 计算链行距：32 的倍数（Cast 与 DataCopyPad 强制 src/dst 32B 对齐）
    int64_t stride = align32(nAl);

    // x1 int8 中转块：[mAl][kAl]，按 int32 尺寸分配（float/int32 视图同尺寸）
    uint32_t x1Bytes = alignUp32(static_cast<uint32_t>(mAl * kAl * static_cast<int64_t>(sizeof(int32_t))));
    if (x1Bytes < MIN_BUF) { x1Bytes = MIN_BUF; }
    uint32_t x1hBytes = alignUp32(static_cast<uint32_t>(mAl * kAl * static_cast<int64_t>(sizeof(half))));
    if (x1hBytes < MIN_BUF) { x1hBytes = MIN_BUF; }

    // x2 int8 中转块：[kAl][stride]
    uint32_t x2Bytes = alignUp32(static_cast<uint32_t>(kAl * stride * static_cast<int64_t>(sizeof(int32_t))));
    if (x2Bytes < MIN_BUF) { x2Bytes = MIN_BUF; }
    uint32_t x2hBytes = alignUp32(static_cast<uint32_t>(kAl * stride * static_cast<int64_t>(sizeof(half))));
    if (x2hBytes < MIN_BUF) { x2hBytes = MIN_BUF; }

    // acc int32 累加块 [mAl][stride]
    uint32_t accBytes = alignUp32(static_cast<uint32_t>(mAl * stride * static_cast<int64_t>(sizeof(int32_t))));
    if (accBytes < MIN_BUF) { accBytes = MIN_BUF; }
    // tmp：MatmulBlock 的 int32 临时 / DequantBlock 的 per-row float，均 nAl<=stride
    uint32_t tmpBytes = alignUp32(static_cast<uint32_t>(stride * static_cast<int64_t>(sizeof(int32_t))));
    if (tmpBytes < MIN_BUF) { tmpBytes = MIN_BUF; }

    uint32_t scaleBytes = alignUp32(static_cast<uint32_t>(stride * static_cast<int64_t>(sizeof(float))));
    if (scaleBytes < MIN_BUF) { scaleBytes = MIN_BUF; }
    uint32_t pertokenBytes = alignUp32(static_cast<uint32_t>(mAl * static_cast<int64_t>(sizeof(float))));
    if (pertokenBytes < MIN_BUF) { pertokenBytes = MIN_BUF; }

    uint32_t outBytes = alignUp32(static_cast<uint32_t>(mAl * stride * static_cast<int64_t>(sizeof(bfloat16_t))));
    if (outBytes < MIN_BUF) { outBytes = MIN_BUF; }

    pipe.InitBuffer(x1Buf, x1Bytes);
    pipe.InitBuffer(x1hBuf, x1hBytes);
    pipe.InitBuffer(x1fBuf, x1Bytes);
    pipe.InitBuffer(x2Buf, x2Bytes);
    pipe.InitBuffer(x2hBuf, x2hBytes);
    pipe.InitBuffer(x2fBuf, x2Bytes);
    pipe.InitBuffer(accBuf, accBytes);
    pipe.InitBuffer(tmpBuf, tmpBytes);
    pipe.InitBuffer(scaleBuf, scaleBytes);
    pipe.InitBuffer(pertokenBuf, pertokenBytes);
    pipe.InitBuffer(outBuf, outBytes);
}

// 逐行一维 burst 从 GM 搬入 UB，规避二维 DataCopy 的 srcStride/dstStride
// 单位为 32B 块（非字节）导致的错位；行距统一取 32 倍数保证 32B 对齐。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadX1Block(int64_t mBase, int64_t kBase, int64_t mCnt, int64_t kCnt, int64_t kAl)
{
    LocalTensor<int8_t> x1Raw = x1Buf.Get<int8_t>();
    DataCopyExtParams sp{1, static_cast<uint32_t>(kCnt), 0, 0, 0};
    DataCopyPadExtParams<int8_t> padInt8{false, 0, static_cast<uint8_t>(kAl - kCnt), static_cast<int8_t>(0)};
    for (int64_t m = 0; m < mCnt; ++m) {
        DataCopyPad(x1Raw[static_cast<uint32_t>(m * kAl)],
            inputGMX[static_cast<uint32_t>((mBase + m) * K_ + kBase)], sp, padInt8);
    }
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadX2Block(int64_t kBase, int64_t nBase, int64_t kCnt, int64_t nCnt, int64_t stride)
{
    LocalTensor<int8_t> x2Raw = x2Buf.Get<int8_t>();
    DataCopyExtParams sp{1, static_cast<uint32_t>(nCnt), 0, 0, 0};
    DataCopyPadExtParams<int8_t> padInt8{false, 0, static_cast<uint8_t>(stride - nCnt), static_cast<int8_t>(0)};
    for (int64_t k = 0; k < kCnt; ++k) {
        DataCopyPad(x2Raw[static_cast<uint32_t>(k * stride)],
            inputGMX2[static_cast<uint32_t>((kBase + k) * N_ + nBase)], sp, padInt8);
    }
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadScaleBlock(int64_t nBase, int64_t nCnt, int64_t stride)
{
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    // 清零到 stride，保证 [nCnt..nAl) 补 0，避免未初始化 UB 参与 Mul
    Duplicate(scaleLocal, 0.0f, static_cast<int32_t>(stride));
    DataCopyExtParams sp{1, static_cast<uint32_t>(nCnt * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padF{false, 0, 0, 0.0f};
    DataCopyPad(scaleLocal, scaleGM[static_cast<uint32_t>(nBase)], sp, padF);
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadPertokenBlock(int64_t mBase, int64_t mCnt)
{
    LocalTensor<float> pertokenLocal = pertokenBuf.Get<float>();
    for (int64_t m = 0; m < mCnt; ++m) {
        pertokenLocal.SetValue(static_cast<uint32_t>(m), pertokenScaleGM.GetValue(static_cast<uint32_t>(mBase + m)));
    }
}

// ======================================================================
// MatmulBlock：int8 -> half -> int32 精确累加（K 很大时 float 累加会丢精度）
// ======================================================================
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::MatmulBlock(int64_t mCnt, int64_t kCnt, int64_t kAl, int64_t stride)
{
    LocalTensor<int8_t> x1Raw = x1Buf.Get<int8_t>();
    LocalTensor<int8_t> x2Raw = x2Buf.Get<int8_t>();
    LocalTensor<half> x1h = x1hBuf.Get<half>();
    LocalTensor<half> x2h = x2hBuf.Get<half>();
    LocalTensor<int32_t> x1f = x1fBuf.Get<int32_t>();
    LocalTensor<int32_t> x2f = x2fBuf.Get<int32_t>();
    LocalTensor<int32_t> acc = accBuf.Get<int32_t>();

    // stride 已是 32 的倍数（Process 中 (nAl+31)&~31L 得出），直接作为向量指令宽度
    int32_t nAl = static_cast<int32_t>(stride);

    // int8 -> half 用 CAST_NONE（合法）；half -> int32 必须用 CAST_RINT
    // （dav_c220 的 half->int32 不支持 CAST_NONE，曾致仿真器 abort 后假 PASS）。
    // int8 值域 [-128,127] 在 half 中精确表示，CAST_RINT 取整无损。
    Cast(x1h, x1Raw, RoundMode::CAST_NONE, static_cast<int32_t>(mCnt * kAl));
    Cast(x1f, x1h, RoundMode::CAST_RINT, static_cast<int32_t>(mCnt * kAl));
    Cast(x2h, x2Raw, RoundMode::CAST_NONE, static_cast<int32_t>(kCnt * stride));
    Cast(x2f, x2h, RoundMode::CAST_RINT, static_cast<int32_t>(kCnt * stride));

    // Cast 走 V 管道，GetValue 走 S 管道。缺 V_S 同步时 Scalar 会读到
    // 尚未落地的 UB 数据，真机上表现为 ±1 ULP 非确定性抖动（仿真器
    // 串行执行无法暴露）。参照 QMSQ LocalAbsMax 的 V_S 惯用法。
    {
        event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evVS);
        WaitFlag<HardEvent::V_S>(evVS);
    }

    // 外积：acc[m][n] += x1[m,k]*x2[k][n]，int32 精确累加。
    // Axpy 在 dav_c220 不支持 int32（编译错误），故用 Muls+Add 两步；
    // tmp 必须作中转，不可原地 Muls(x2Row, x2Row, ...)（会污染 x2f，下一行 m 读到
    // 已被乘过的数据，M>1 时 100% 错，实测 bad=3840/120）。
    // 不可把 x1 标量预取到寄存器数组：实测导致编译器把 GetValue 结果溢出到 UB
    // 反复读写，case a（K=32768）回退 33%（2364→3168μs）。
    LocalTensor<int32_t> tmp = tmpBuf.Get<int32_t>();
    for (int64_t m = 0; m < mCnt; ++m) {
        LocalTensor<int32_t> accRow = acc[static_cast<uint32_t>(m * stride)];
        for (int64_t k = 0; k < kCnt; ++k) {
            int32_t a = x1f.GetValue(static_cast<uint32_t>(m * kAl + k));
            // GetValue 走 S 管道，Muls/Add 走 V 管道，需 S_V 同步确保 a 已就绪
            event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evSV);
            WaitFlag<HardEvent::S_V>(evSV);
            Muls(tmp, x2f[static_cast<uint32_t>(k * stride)], a, nAl);
            Add(accRow, accRow, tmp, nAl);
        }
    }
}

// ======================================================================
// DequantBlock：acc_int32 -> fp32 -> *scale(per-col) -> *pertoken(per-row) -> bf16
// ======================================================================
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::DequantBlock(int64_t mCnt, int64_t nCnt, int64_t stride)
{
    LocalTensor<int32_t> acc = accBuf.Get<int32_t>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> pertokenLocal = pertokenBuf.Get<float>();
    LocalTensor<bfloat16_t> outB = outBuf.Get<bfloat16_t>();
    LocalTensor<float> rowF = tmpBuf.Get<float>();

    int32_t nAl = static_cast<int32_t>((nCnt + 7) & ~7L);
    if (nAl > stride) { nAl = static_cast<int32_t>(stride); }

    for (int64_t m = 0; m < mCnt; ++m) {
        LocalTensor<int32_t> accRowI = acc[static_cast<uint32_t>(m * stride)];
        // int32 -> float（整数值精确），× scale（per-col 向量）× pertoken（per-row 标量）
        Cast(rowF, accRowI, RoundMode::CAST_NONE, nAl);
        Mul(rowF, rowF, scaleLocal, nAl);
        Muls(rowF, rowF, pertokenLocal.GetValue(static_cast<uint32_t>(m)), nAl);
        // float -> bf16，CAST_RINT 与 numpy/ml_dtypes bfloat16 舍入语义一致
        Cast(outB[static_cast<uint32_t>(m * stride)], rowF, RoundMode::CAST_RINT, nAl);
    }
}

// 逐行一维 burst 写出 GM：blockCount=1, blockLen=nCnt*2B, srcStride=dstStride=0。
// 不用二维 burst：其 srcStride/dstStride 单位为 32B 块（非字节），int8/bf16
// 行距无法正确表达，曾致 100% Wrong Answer（参照 QMSQ 已确证结论）。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::WriteOutBlock(int64_t mBase, int64_t nBase, int64_t mCnt, int64_t nCnt, int64_t stride)
{
    LocalTensor<bfloat16_t> outB = outBuf.Get<bfloat16_t>();
    for (int64_t m = 0; m < mCnt; ++m) {
        DataCopyExtParams yp{1, static_cast<uint32_t>(nCnt * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(outGM[static_cast<uint32_t>((mBase + m) * N_ + nBase)], outB[static_cast<uint32_t>(m * stride)], yp);
    }
}

// ======================================================================
// public API
// ======================================================================
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, const QuantBatchMatMulV3TilingData* tilingData)
{
    ParseTiling(tilingData);

    blockNum_ = static_cast<int64_t>(GetBlockNum());
    blockIdx_ = static_cast<int64_t>(GetBlockIdx());
    if (blockNum_ < 1) { blockNum_ = 1; }
    if (blockIdx_ < 0) { blockIdx_ = 0; }

    SetupGlobal(x1, x2, scale, pertokenScale, out);
    AllocBuffers();
}

// 多核按 M 块轮询切分（m = blockIdx*chunkM, +blockNum*chunkM, ...）：
// 连续段分配在 blockDim < totalMChunks 时会漏掉尾部行，必须轮询覆盖全部块。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Process()
{
    for (int64_t m = blockIdx_ * mCnt_; m < M_; m += blockNum_ * mCnt_) {
        int64_t mCnt = M_ - m;
        if (mCnt > mCnt_) { mCnt = mCnt_; }
        if (mCnt <= 0) { break; }
        int64_t mAl = (mCnt + 7) & ~7L;

        LoadPertokenBlock(m, mCnt);

        for (int64_t n = 0; n < N_; n += chunkN_) {
            int64_t nCnt = N_ - n;
            if (nCnt > chunkN_) { nCnt = chunkN_; }
            if (nCnt <= 0) { break; }
            int64_t nAl = (nCnt + 7) & ~7L;
            int64_t stride = (nAl + 31) & ~31L;

            LoadScaleBlock(n, nCnt, stride);

            // 累加器清零（覆盖 mAl 行 × stride 列）
            {
                LocalTensor<int32_t> acc = accBuf.Get<int32_t>();
                Duplicate(acc, static_cast<int32_t>(0), static_cast<int32_t>(mAl * stride));
            }

            // K 维分块累加
            for (int64_t k = 0; k < K_; k += kBlk_) {
                int64_t kCnt = K_ - k;
                if (kCnt > kBlk_) { kCnt = kBlk_; }
                if (kCnt <= 0) { break; }
                int64_t kAl = (kCnt + 31) & ~31L;
                LoadX1Block(m, k, mCnt, kCnt, kAl);
                LoadX2Block(k, n, kCnt, nCnt, stride);
                // LoadX*Block 走 MTE2 管道，MatmulBlock 的 Cast 读 UB 走 V 管道，
                // 缺 MTE2_V 同步会读到未搬完的 GM 数据（真机非确定性错）
                {
                    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(ev);
                    WaitFlag<HardEvent::MTE2_V>(ev);
                }
                MatmulBlock(mCnt, kCnt, kAl, stride);
            }

            DequantBlock(mCnt, nCnt, stride);
            // DequantBlock 的 Cast（V）写 outB，WriteOutBlock 的 DataCopyPad（MTE3）读，
            // 缺 V_MTE3 同步会写出未完成计算的 bf16
            {
                event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(ev);
                WaitFlag<HardEvent::V_MTE3>(ev);
            }
            WriteOutBlock(m, n, mCnt, nCnt, stride);
            // 写 GM 完成后再进下一轮 n 块，避免覆盖未读走的 outB / acc
            {
                event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(ev);
                WaitFlag<HardEvent::MTE3_MTE2>(ev);
            }
        }
    }

    // 确保所有 MTE3 写 GM 完成后再退出
    event_t evEnd = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(evEnd);
    WaitFlag<HardEvent::MTE3_MTE2>(evEnd);
}

} // namespace NsQuantBatchMatMulV3
#endif // QUANTBATCHMATMULV3_H
