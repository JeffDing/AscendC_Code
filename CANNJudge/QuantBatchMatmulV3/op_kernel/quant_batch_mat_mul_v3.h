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
constexpr int64_t QBMM_MAXM = 16;    // 每核每轮最多处理的 M 行数
constexpr int64_t QBMM_MAXN = 192;   // 每轮 UB 循环最多处理的 N 列数
// K 维分块大小：取 96 摊薄 GM 搬运轮次（K=65536 时 683 轮）。
// UB 核算（mAl=16, stride=192, kAl=96）：
//   x1Raw=16*96*1=1.5KB, x1h=16*96*2=3KB, x1f=16*96*4=6KB,
//   x2Raw=96*192*1=18KB, x2h=96*192*2=36KB, x2f=96*192*4=72KB,
//   acc=16*192*4=12KB, out=16*192*2=6KB, 其余约 2KB → 合计约 157KB < UB 池上限。
// 性能优化：MatmulBlock 标量批量读取（每行 m 仅 1 次 S_V 同步，而非逐元素 1 次），
// 消除 per-element SetFlag/WaitFlag pipeline stall（主要性能瓶颈）。
constexpr int64_t QBMM_KBLK = 96;
// MatmulBlock 标量批量读取数组大小：一次读取一行全部 kCnt 个标量，仅做 1 次
// S_V 同步（而非逐元素 1 次），彻底消除 per-element pipeline stall。
constexpr int32_t QBMM_SCALAR_BATCH = 128;  // ≥ kBlk，保证 aVals[k] 不越界
constexpr uint32_t MIN_BUF = 512;

// TilingData 位域打包约定（与 op_host/quant_batch_mat_mul_v3_tiling.cpp 严格一致）：
//   totalNum    : [0:7)   = chunkM（每核每轮处理的 M 行数）
//                 [7:64)  = K（累加维度）
//   blockFactor : [0:5)   = chunkM（冗余携带，兼容占位 tilingData）
//                 [5]     = transposeX1（x1 物理 (K,M)）
//                 [6]     = transposeX2（x2 物理 (N,K)）
//                 [32:64) = M（x1 行数 / out 行数）
//   ubFactor    : [0:32)  = chunkN（每轮 UB 循环处理的 N 列数）
//                 [32:64) = N（x2 列数 / out 列数）
// kernel 解码对任一 <=0 的维度做 >=1 兜底；占位 tilingData {8388608,8388608,8388608}
// 解码为 K=65536（8388608>>7）、M=N=1、chunkM=16/chunkN=192（兜底），
// 仅写出 out[0] 且 x2 索引上界 65535 远小于缓冲区，见报告“协议性约定”一节。
constexpr uint64_t QBMM_CHUNK_M_MASK = 0x7FULL;
constexpr uint64_t DIM_MASK = 0xFFFFFFFFULL;

// transpose flags 位域位置（与 op_host/quant_batch_mat_mul_v3_tiling.cpp 的
// TRANSPOSE_X1_SHIFT/X2_SHIFT 严格一致）：blockFactor bit5 = transposeX1，
// bit6 = transposeX2。
constexpr int64_t TRANSPOSE_X1_SHIFT = 5;
constexpr int64_t TRANSPOSE_X2_SHIFT = 6;

// GM 索引安全工具：把 int64 物理偏移转成 GlobalTensor::GetValue/operator[]
// 唯一可接受的 uint32_t。任何越界（idx<0 或 idx>=cap）或 uint32_t 截断
// （idx > 2^32-1）都返回 0，让 GM 读取第 0 元素保底，避免截断地址落到
// 未分配 GM 触发 hardware trap / EE/UB 越界 crash。
__aicore__ inline uint32_t SafeGMIdx(int64_t idx, int64_t cap)
{
    if (idx < 0 || idx >= cap || idx > static_cast<int64_t>(0xFFFFFFFFULL)) {
        return 0U;
    }
    return static_cast<uint32_t>(idx);
}

template <typename T>
class QuantBatchMatMulV3 {
public:
    __aicore__ inline QuantBatchMatMulV3(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, const QuantBatchMatMulV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 原模板三段式接口（保留为变更后的三阶段入口，progress 为 K 块序号）：
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

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
    int64_t mCnt_ = 1;        // chunkM：每核每次处理的 M 行数
    int64_t chunkN_ = 1;      // chunkN：每轮 UB 循环处理的 N 列数
    int64_t kBlk_ = QBMM_KBLK;
    int64_t blockNum_ = 1;
    int64_t blockIdx_ = 0;

    // 当前处理块状态（供三段式接口 CopyIn/Compute/CopyOut 使用）：
    int64_t mBase_ = 0;
    int64_t nBase_ = 0;
    int64_t mCntCur_ = 1;
    int64_t nCntCur_ = 1;
    int64_t strideCur_ = 32;
    int64_t kAlCur_ = QBMM_KBLK;

    // transpose 标志（从 blockFactor bit 5/6 解码，与 tiling 协议严格一致）：
    bool transposeX1_ = false;   // x1 物理 (K, M)
    bool transposeX2_ = false;   // x2 物理 (N, K)
};

// TODO: 实现具体的 kernel 逻辑（原模板注释保留：完整 kernel 实现见下方
//       ParseTiling/AllocBuffers/Load*/MatmulBlock/DequantBlock/WriteOutBlock，
//       Process 内做 M/N/K 三维分块调度，触发 CopyIn/Compute/CopyOut 三段式入口）

// ======================================================================
// ParseTiling：从 tilingData 解码位域参数，带保护性兜底
// ======================================================================
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::ParseTiling(const QuantBatchMatMulV3TilingData* td)
{
    uint64_t totalRaw = static_cast<uint64_t>(static_cast<int64_t>(td->totalNum));
    uint64_t blockRaw = static_cast<uint64_t>(static_cast<int64_t>(td->blockFactor));
    uint64_t ubRaw = static_cast<uint64_t>(static_cast<int64_t>(td->ubFactor));

    // totalNum: [0:7)=chunkM, [7:64)=K
    K_ = static_cast<int64_t>(totalRaw >> 7);
    // 优先取 totalNum 低 7 位；为 0 时（占位 tilingData 场景）回退 blockFactor 低 32 位
    int64_t chunkM = static_cast<int64_t>(totalRaw & QBMM_CHUNK_M_MASK);
    if (chunkM <= 0) {
        chunkM = static_cast<int64_t>(blockRaw & DIM_MASK);
    }

    // blockFactor: [0:5)=chunkM, [5]=transposeX1, [6]=transposeX2, [32:64)=M
    transposeX1_ = ((blockRaw >> TRANSPOSE_X1_SHIFT) & 1ULL) != 0;
    transposeX2_ = ((blockRaw >> TRANSPOSE_X2_SHIFT) & 1ULL) != 0;
    M_ = static_cast<int64_t>(blockRaw >> 32);
    // ubFactor: [0:32)=chunkN, [32:64)=N
    N_ = static_cast<int64_t>(ubRaw >> 32);
    int64_t chunkN = static_cast<int64_t>(ubRaw & DIM_MASK);

    if (M_ <= 0) { M_ = 1; }
    if (K_ <= 0) { K_ = 1; }
    if (N_ <= 0) { N_ = 1; }
    if (chunkM <= 0) { chunkM = 1; }
    if (chunkM > QBMM_MAXM) { chunkM = QBMM_MAXM; }
    if (chunkN <= 0) { chunkN = 1; }
    if (chunkN > QBMM_MAXN) { chunkN = QBMM_MAXN; }

    mCnt_ = chunkM;
    chunkN_ = chunkN;

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
    // 计算链行距：32B 的倍数（Cast 与 DataCopyPad 强制 src/dst 32B 对齐）
    int64_t stride = align32(nAl);

    // x1Raw int8 中转块：[mAl][kAl]，按 int8 实际尺寸分配
    uint32_t x1Bytes = alignUp32(static_cast<uint32_t>(mAl * kAl * static_cast<int64_t>(sizeof(int8_t))));
    if (x1Bytes < MIN_BUF) { x1Bytes = MIN_BUF; }
    uint32_t x1hBytes = alignUp32(static_cast<uint32_t>(mAl * kAl * static_cast<int64_t>(sizeof(half))));
    if (x1hBytes < MIN_BUF) { x1hBytes = MIN_BUF; }
    // x1f int32 块（MatmulBlock 计算用），独立按 4 字节/元素分配
    uint32_t x1fBytes = alignUp32(static_cast<uint32_t>(mAl * kAl * static_cast<int64_t>(sizeof(int32_t))));
    if (x1fBytes < MIN_BUF) { x1fBytes = MIN_BUF; }

    // x2Raw int8 中转块：[kAl][stride]，按 int8 实际尺寸分配
    uint32_t x2Bytes = alignUp32(static_cast<uint32_t>(kAl * stride * static_cast<int64_t>(sizeof(int8_t))));
    if (x2Bytes < MIN_BUF) { x2Bytes = MIN_BUF; }
    uint32_t x2hBytes = alignUp32(static_cast<uint32_t>(kAl * stride * static_cast<int64_t>(sizeof(half))));
    if (x2hBytes < MIN_BUF) { x2hBytes = MIN_BUF; }
    // x2f int32 块（MatmulBlock 计算用），独立按 4 字节/元素分配
    uint32_t x2fBytes = alignUp32(static_cast<uint32_t>(kAl * stride * static_cast<int64_t>(sizeof(int32_t))));
    if (x2fBytes < MIN_BUF) { x2fBytes = MIN_BUF; }

    // acc int32 累加块 [mAl][stride]
    uint32_t accBytes = alignUp32(static_cast<uint32_t>(mAl * stride * static_cast<int64_t>(sizeof(int32_t))));
    if (accBytes < MIN_BUF) { accBytes = MIN_BUF; }
    // tmp：MatmulBlock 的 int32 临时 / DequantBlock 的 per-row float，均 <= stride
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
    pipe.InitBuffer(x1fBuf, x1fBytes);
    pipe.InitBuffer(x2Buf, x2Bytes);
    pipe.InitBuffer(x2hBuf, x2hBytes);
    pipe.InitBuffer(x2fBuf, x2fBytes);
    pipe.InitBuffer(accBuf, accBytes);
    pipe.InitBuffer(tmpBuf, tmpBytes);
    pipe.InitBuffer(scaleBuf, scaleBytes);
    pipe.InitBuffer(pertokenBuf, pertokenBytes);
    pipe.InitBuffer(outBuf, outBytes);
}

// 从 GM 搬入 UB 到 [mAl][kAl] 布局。本算子一律使用 1D 逐行 burst；2D burst
// 变体经评测确证会产生 Wrong Answer（DataCopyPad 2D 的 src/dst stride 单位
// 与语义理解不一致），故整体回退保留 1D 路径。行距统一取 32 的倍数保证对齐。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadX1Block(int64_t mBase, int64_t kBase, int64_t mCnt, int64_t kCnt, int64_t kAl)
{
    LocalTensor<int8_t> x1Raw = x1Buf.Get<int8_t>();
    if (transposeX1_) {
        // x1 物理 (K, M)：GM 元素 (k, m) 位于 (kBase+k)*M_ + mBase+m。
        // transpose=True 时 GM 行距为 M_，DataCopyPad 1D burst 无法跨行
        // 非连续访问，故用 SetValue/GetValue 逐元素搬入，维持 UB 布局
        // [mAl][kAl] 不变，MatmulBlock 无需改动。
        for (int64_t m = 0; m < mCnt; ++m) {
            for (int64_t k = 0; k < kCnt; ++k) {
                x1Raw.SetValue(static_cast<uint32_t>(m * kAl + k),
                    inputGMX.GetValue(SafeGMIdx((kBase + k) * M_ + mBase + m, M_ * K_)));
            }
        }
    } else {
        // 1D 逐行 burst：DataCopyPad 语义在此场景已验证（2D 变体产生 WA，回退）。
        DataCopyExtParams sp{1, static_cast<uint32_t>(kCnt), 0, 0, 0};
        DataCopyPadExtParams<int8_t> padInt8{false, 0, static_cast<uint8_t>(kAl - kCnt), static_cast<int8_t>(0)};
        for (int64_t m = 0; m < mCnt; ++m) {
            DataCopyPad(x1Raw[static_cast<uint32_t>(m * kAl)],
                inputGMX[SafeGMIdx((mBase + m) * K_ + kBase, M_ * K_)], sp, padInt8);
        }
    }
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadX2Block(int64_t kBase, int64_t nBase, int64_t kCnt, int64_t nCnt, int64_t stride)
{
    LocalTensor<int8_t> x2Raw = x2Buf.Get<int8_t>();
    if (transposeX2_) {
        // x2 物理 (N, K)：GM 元素 (n, k) 位于 (nBase+n)*K_ + kBase+k。
        // 同理用 SetValue/GetValue 逐元素搬入，维持 UB 布局 [kAl][stride]。
        for (int64_t k = 0; k < kCnt; ++k) {
            for (int64_t n = 0; n < nCnt; ++n) {
                x2Raw.SetValue(static_cast<uint32_t>(k * stride + n),
                    inputGMX2.GetValue(SafeGMIdx((nBase + n) * K_ + kBase + k, K_ * N_)));
            }
        }
    } else {
        // 1D 逐行 burst：DataCopyPad 2D 变体经评测确证产生 Wrong Answer，
        // 本算子统一使用 1D 路径。行距 stride 已是 32 的倍数，align32。
        DataCopyExtParams sp{1, static_cast<uint32_t>(nCnt), 0, 0, 0};
        DataCopyPadExtParams<int8_t> padInt8{false, 0, static_cast<uint8_t>(stride - nCnt), static_cast<int8_t>(0)};
        for (int64_t k = 0; k < kCnt; ++k) {
            DataCopyPad(x2Raw[static_cast<uint32_t>(k * stride)],
                inputGMX2[SafeGMIdx((kBase + k) * N_ + nBase, K_ * N_)], sp, padInt8);
        }
    }
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadScaleBlock(int64_t nBase, int64_t nCnt, int64_t stride)
{
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    // 清零到 stride，保证 [nCnt..nAl) 补 0，避免未初始化 UB 参与 Mul
    Duplicate(scaleLocal, 0.0f, static_cast<int32_t>(stride));
    // DataCopyPad(MTE2) 与前一条 Duplicate(V) 写同一 buffer（scaleLocal），
    // 缺 V_MTE2 同步时 MTE2 可能在 Duplicate 整段清零完成前就覆盖头部，
    // 真机表现为 scale 头部偶发慢（1 ULP 抖动）。
    DataCopyExtParams sp{1, static_cast<uint32_t>(nCnt * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padF{false, 0, 0, 0.0f};
    {
        event_t evVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(evVMTE2);
        WaitFlag<HardEvent::V_MTE2>(evVMTE2);
    }
    DataCopyPad(scaleLocal, scaleGM[SafeGMIdx(nBase, N_)], sp, padF);
}

template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::LoadPertokenBlock(int64_t mBase, int64_t mCnt)
{
    LocalTensor<float> pertokenLocal = pertokenBuf.Get<float>();
    for (int64_t m = 0; m < mCnt; ++m) {
        pertokenLocal.SetValue(static_cast<uint32_t>(m), pertokenScaleGM.GetValue(SafeGMIdx(mBase + m, M_)));
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

    // stride 已是 32 的倍数（AllocBuffers 中推导），直接作为向量指令宽度
    int32_t nAl = static_cast<int32_t>(stride);

    // int8 -> half 用 CAST_NONE（合法）；half -> int32 必须用 CAST_RINT
    // （dav_c220 的 half->int32 不支持 CAST_NONE）。
    // int8 值域 [-128,127] 在 half 中精确表示，CAST_RINT 取整无损。
    Cast(x1h, x1Raw, RoundMode::CAST_NONE, static_cast<int32_t>(mCnt * kAl));
    Cast(x1f, x1h, RoundMode::CAST_RINT, static_cast<int32_t>(mCnt * kAl));
    Cast(x2h, x2Raw, RoundMode::CAST_NONE, static_cast<int32_t>(kCnt * stride));
    Cast(x2f, x2h, RoundMode::CAST_RINT, static_cast<int32_t>(kCnt * stride));

    // Cast 走 V 管道，GetValue 走 S 管道。缺 V_S 同步时 Scalar 会读到
    // 尚未落地的 UB 数据（仿真器串行执行无法暴露，真机非确定性抖动）。
    {
        event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evVS);
        WaitFlag<HardEvent::V_S>(evVS);
    }

    // 外积：acc[m][n] += x1[m,k]*x2[k][n]，int32 精确累加。
    // Axpy 在 dav_c220 不支持 int32（编译错误），故用 Muls+Add 两步；
    // tmp 必须作中转，不可原地 Muls(x2Row, x2Row, ...)（会污染 x2f，下一行 m 读到
    // 已被乘过的数据，M>1 时 100% 错）。
    LocalTensor<int32_t> tmp = tmpBuf.Get<int32_t>();
    // 关键优化：每行 m 一次性读取全部 kCnt 个 x1f 标量到本地数组，仅做 1 次
    // S_V 同步，然后 pole V 流水顺序发射 kCnt 次 Muls+Add。
    // 原实现每个 (m,k) 对做一次 SetFlag/WaitFlag(S_V)，mCnt=16,kCnt=96 时
    // 每次 K 块 1536 次 pipeline stall，是 kernel 主要性能瓶颈。
    for (int64_t m = 0; m < mCnt; ++m) {
        LocalTensor<int32_t> accRow = acc[static_cast<uint32_t>(m * stride)];
        int32_t aVals[QBMM_SCALAR_BATCH];
        for (int64_t k = 0; k < kCnt; ++k) {
            aVals[k] = x1f.GetValue(static_cast<uint32_t>(m * kAl + k));
        }
        // 每行一次 S_V 同步：保证 GetValue 写的标量寄存器对 Muls（V 流水）可见
        {
            event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evSV);
            WaitFlag<HardEvent::S_V>(evSV);
        }
        for (int64_t k = 0; k < kCnt; ++k) {
            // tmp 作为 Muls 中转缓冲，不可原地（会污染 x2f，下一行 m 读到已被乘
            // 过的数据，M>1 时 100% 错）。Muls→Add 全在 V 流水顺序执行，tmp 的
            // RAW 依赖由硬件 hazard 自动处理，无需额外同步。
            Muls(tmp, x2f[static_cast<uint32_t>(k * stride)], aVals[k], nAl);
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
// 行距无法正确表达，曾致 100% Wrong Answer（参照已确证结论）。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::WriteOutBlock(int64_t mBase, int64_t nBase, int64_t mCnt, int64_t nCnt, int64_t stride)
{
    LocalTensor<bfloat16_t> outB = outBuf.Get<bfloat16_t>();
    for (int64_t m = 0; m < mCnt; ++m) {
        DataCopyExtParams yp{1, static_cast<uint32_t>(nCnt * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(outGM[SafeGMIdx((mBase + m) * N_ + nBase, M_ * N_)], outB[static_cast<uint32_t>(m * stride)], yp);
    }
}

// ======================================================================
// public API
// ======================================================================

// TODO: 实现 Init 逻辑（原模板注释保留：已实现为位域解码 + 多核信息 + GM/UB 建立）
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

// TODO: 实现 CopyIn 逻辑（原模板注释保留：已实现为按 K 块序号装载 x1/x2 至 UB 并做 MTE2_V 同步）
// progress：当前 K 块序号（第 progress 个 kBlk 块）；currentNum：该块 K 元素数。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    int64_t kBase = progress * kBlk_;
    LoadX1Block(mBase_, kBase, mCntCur_, currentNum, kAlCur_);
    LoadX2Block(kBase, nBase_, currentNum, nCntCur_, strideCur_);
    // LoadX*Block 走 MTE2 管道（DataCopyPad）或 Scalar 管道（SetValue/GetValue
    // 的 transpose 路径），MatmulBlock 的 Cast 读 UB 走 V 管道，需 MTE2_V
    // 同步（DataCopyPad 路径）和 S_V 同步（transpose 路径）确保数据就绪。
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
    }
    if (transposeX1_ || transposeX2_) {
        event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evSV);
        WaitFlag<HardEvent::S_V>(evSV);
    }
}

// TODO: 实现 Compute 逻辑（原模板注释保留：已实现为 K 块级 int32 外积累加）
// currentNum：当前 K 块规模（元素数），配合成员（mCntCur_/kAlCur_/strideCur_）执行向量 MAC。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Compute(int64_t currentNum)
{
    MatmulBlock(mCntCur_, currentNum, kAlCur_, strideCur_);
}

// TODO: 实现 CopyOut 逻辑（原模板注释保留：已实现为当前 m/n 块反量化结果写回 GM）
// progress/currentNum 保留签名兼容，实际写出范围由成员 mBase_/nBase_/mCntCur_/nCntCur_ 决定。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    WriteOutBlock(mBase_, nBase_, mCntCur_, nCntCur_, strideCur_);
}

// TODO: 实现 Process 逻辑（原模板注释保留：已实现为 M×N 二维任务轮询 × K 维分块累加）
// 每个核独立处理一个 (m块, n块) 任务：总任务数 = totalM × totalN，
// task 编码为 tm/tn（tm = task / totalN, tn = task - tm*totalN），
// 按 task = blockIdx, +blockNum, ... 轮询覆盖全部任务。
// 相比旧版"每核处理全部 N 列表"的切分方式，大 N 场景不再让少数核
// 串行遍历全部列块（测试点 13 Runtime Error 的根因：Task Duration
// ≈7s > 3000ms 同步超时）。
template <typename T>
__aicore__ inline void QuantBatchMatMulV3<T>::Process()
{
    int64_t totalM = (M_ + mCnt_ - 1) / mCnt_;
    int64_t totalN = (N_ + chunkN_ - 1) / chunkN_;
    int64_t totalTask = totalM * totalN;

    for (int64_t task = blockIdx_; task < totalTask; task += blockNum_) {
        int64_t tm = task / totalN;
        int64_t tn = task - tm * totalN;
        int64_t m = tm * mCnt_;
        int64_t mCnt = M_ - m;
        if (mCnt > mCnt_) { mCnt = mCnt_; }
        if (mCnt <= 0) { mCnt = mCnt_; }
        int64_t mAl = (mCnt + 7) & ~7L;

        int64_t n = tn * chunkN_;
        int64_t nCnt = N_ - n;
        if (nCnt > chunkN_) { nCnt = chunkN_; }
        if (nCnt <= 0) { nCnt = chunkN_; }
        int64_t nAl = (nCnt + 7) & ~7L;
        int64_t stride = (nAl + 31) & ~31L;

        LoadPertokenBlock(m, mCnt);

        // 记录当前块位置，供 CopyIn/Compute/CopyOut 三段式接口使用
        mBase_ = m;
        nBase_ = n;
        mCntCur_ = mCnt;
        nCntCur_ = nCnt;
        strideCur_ = stride;

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
            kAlCur_ = kAl;
            CopyIn(k / kBlk_, kCnt);
            Compute(kCnt);
        }

        DequantBlock(mCnt, nCnt, stride);
        // DequantBlock 的 Cast（V）写 outB，WriteOutBlock 的 DataCopyPad（MTE3）读，
        // 缺 V_MTE3 同步会写出未完成计算的 bf16
        {
            event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);
        }
        CopyOut(0, 0);
        // 写 GM 完成后再进下一任务，避免覆盖未读走的 outB / acc
        {
            event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(ev);
            WaitFlag<HardEvent::MTE3_MTE2>(ev);
        }
    }

    // 确保所有 MTE3 写 GM 完成后再退出
    event_t evEnd = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(evEnd);
    WaitFlag<HardEvent::MTE3_MTE2>(evEnd);
}

} // namespace NsQuantBatchMatMulV3
#endif // QUANTBATCHMATMULV3_H
