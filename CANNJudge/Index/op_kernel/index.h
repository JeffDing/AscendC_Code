/*!
 * \file index.h
 * \brief Index 算子 kernel 类定义
 */

#ifndef INDEX_H
#define INDEX_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "index_tiling_data.h"
#include "index_tiling_key.h"

namespace NsIndex {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MAX_NDIM = 8;
constexpr int64_t DIM_MASK = (int64_t(1) << 20) - 1;
constexpr int64_t Q_MASK = (int64_t(1) << 22) - 1;
constexpr int64_t TAIL_CHUNK_MAX = 4096;
// UB 单次搬运允许的最大元素数（决定块 DMA 的 blockCount 上限）
constexpr int64_t UB_ELEMS_MAX = 4096;
constexpr int64_t IDX_BUF_ELEMS = 8192;

template <typename T>
class Index {
public:
    __aicore__ inline Index(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indexed_sizes, GM_ADDR indices, GM_ADDR y, const IndexTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyInBlocks(int64_t progress, int64_t blkCount, int64_t blkLen, int64_t srcGapBytes);
    __aicore__ inline void CopyOutBlocks(int64_t progress, int64_t blkCount, int64_t blkLen, int64_t dstGapBytes);
    __aicore__ inline void BlockCopy(int64_t src0, int64_t dst0, int64_t blkLen, int64_t srcStep, int64_t dstStep,
                                     int64_t blkCount);

    __aicore__ inline void ParseTiling(const IndexTilingData* tilingData);
    __aicore__ inline void DeriveLayout(GM_ADDR indexed_sizes, GM_ADDR indices);
    __aicore__ inline void Prepare(GM_ADDR x, GM_ADDR indices, GM_ADDR y);
    __aicore__ inline int64_t DecodeOffset(int64_t idx, const int64_t* dimList, int64_t cnt) const;
    __aicore__ inline int64_t ReadInt64(const LocalTensor<int32_t>& buf, int64_t elemIdx) const;
    __aicore__ inline void CopyIndicesChunk(int64_t startElem, int64_t count);
    __aicore__ inline int64_t IndexValue(int64_t t);
    __aicore__ inline void Gather(int64_t srcOff, int64_t dstOff, int64_t len);
    __aicore__ inline void IdentityCopy();
    __aicore__ inline void Drain();
    // 多块 DMA 要求每个数据块在 UB 侧按 32B 对齐，故仅当块字节数为 32B 倍数时才可用块 DMA
    __aicore__ inline bool CanBatch(int64_t blkLen) const
    {
        return ((blkLen * static_cast<int64_t>(sizeof(T)) * byteUnits_) % 32) == 0;
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufB_;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    int32_t curBuf_ = 0;
    bool pending_[2] = {false, false};

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    GlobalTensor<int64_t> indicesGM;

    GM_ADDR indicesAddr_ = nullptr;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;

    int64_t ndim_ = 0;
    int64_t dims_[MAX_NDIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t strides_[MAX_NDIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t q_ = 0;
    // 每个元素占 T 的个数：fp16/fp32 为 1，int64（8B）在 float 模板下为 2
    int64_t byteUnits_ = 1;

    int64_t preDims_[MAX_NDIM] = {0};
    int64_t preCnt_ = 0;
    int64_t preProd_ = 1;
    int64_t outerDims_[MAX_NDIM] = {0};
    int64_t outerCnt_ = 0;
    int64_t outerNum_ = 1;
    int64_t tail_ = 1;
    int64_t preStride_ = 0;
    int64_t maskDims_[MAX_NDIM] = {0};
    int64_t maskCnt_ = 0;
    int64_t rr_ = 1;
    int64_t totalUnits_ = 0;
    int64_t tailChunk_ = 1;
    int64_t xTotal_ = 0;
    int64_t outTotal_ = 0;
    int64_t idxChunk_ = -1;
    bool idxCached_ = false;
    bool identity_ = false;
    bool active_ = false;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Index<T>::Init(GM_ADDR x, GM_ADDR indexed_sizes, GM_ADDR indices, GM_ADDR y, const IndexTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    ParseTiling(tilingData);
    if (!active_) {
        return;
    }
    DeriveLayout(indexed_sizes, indices);
    Prepare(x, indices, y);
}

template <typename T>
__aicore__ inline void Index<T>::ParseTiling(const IndexTilingData* tilingData)
{
    uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);
    uint64_t totalNum = static_cast<uint64_t>(tilingData->totalNum);

    ndim_ = static_cast<int64_t>(blockFactor & 0xFULL);
    if (ndim_ < 1 || ndim_ > MAX_NDIM) {
        active_ = false;
        return;
    }
    dims_[0] = static_cast<int64_t>((blockFactor >> 4) & static_cast<uint64_t>(DIM_MASK));
    dims_[1] = static_cast<int64_t>((blockFactor >> 24) & static_cast<uint64_t>(DIM_MASK));
    dims_[2] = static_cast<int64_t>((blockFactor >> 44) & static_cast<uint64_t>(DIM_MASK));
    dims_[3] = static_cast<int64_t>(ubFactor & static_cast<uint64_t>(DIM_MASK));
    dims_[4] = static_cast<int64_t>((ubFactor >> 20) & static_cast<uint64_t>(DIM_MASK));
    q_ = static_cast<int64_t>((ubFactor >> 40) & static_cast<uint64_t>(Q_MASK));
    dims_[5] = static_cast<int64_t>(totalNum & static_cast<uint64_t>(DIM_MASK));
    dims_[6] = static_cast<int64_t>((totalNum >> 20) & static_cast<uint64_t>(DIM_MASK));
    dims_[7] = static_cast<int64_t>((totalNum >> 40) & static_cast<uint64_t>(DIM_MASK));

    // dtype 字节数编码：[62:64) 0=2B 1=4B 2=8B；换算成“每元素占 T 的个数”
    int64_t dtypeCode = static_cast<int64_t>((ubFactor >> 62) & 0x3ULL);
    int64_t elemBytes = (dtypeCode == 0) ? 2 : ((dtypeCode == 1) ? 4 : 8);
    int64_t tBytes = static_cast<int64_t>(sizeof(T));
    byteUnits_ = elemBytes / tBytes;
    if (byteUnits_ < 1) {
        byteUnits_ = 1;
    }

    for (int64_t i = 0; i < ndim_; ++i) {
        if (dims_[i] <= 0) {
            active_ = false;
            return;
        }
    }
    strides_[ndim_ - 1] = 1;
    for (int64_t i = ndim_ - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
    active_ = true;
}

template <typename T>
__aicore__ inline int64_t Index<T>::ReadInt64(const LocalTensor<int32_t>& buf, int64_t elemIdx) const
{
    uint64_t lo = static_cast<uint64_t>(static_cast<uint32_t>(buf.GetValue(static_cast<int32_t>(elemIdx * 2))));
    uint64_t hi = static_cast<uint64_t>(static_cast<uint32_t>(buf.GetValue(static_cast<int32_t>(elemIdx * 2 + 1))));
    return static_cast<int64_t>((hi << 32) | lo);
}

template <typename T>
__aicore__ inline void Index<T>::DeriveLayout(GM_ADDR indexed_sizes, GM_ADDR indices)
{
    indicesAddr_ = indices;
    pipe.InitBuffer(maskBuf, static_cast<int32_t>(((ndim_ * 8 + 31) / 32) * 32));
    LocalTensor<int32_t> maskLocal = maskBuf.Get<int32_t>();
    GlobalTensor<int32_t> maskGM;
    maskGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indexed_sizes), static_cast<uint64_t>(ndim_ * 2));
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(ndim_ * 8), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, static_cast<int32_t>(0)};
    DataCopyPad(maskLocal, maskGM, copyParams, padParams);
    // MTE2 搬入 UB 后需读标量：用全流水屏障确保标量读（S）在 DMA 完成之后
    PipeBarrier<PIPE_ALL>();

    int64_t maskedDims[MAX_NDIM] = {0};
    int64_t maskedCnt = 0;
    for (int64_t i = 0; i < ndim_; ++i) {
        if (ReadInt64(maskLocal, i) != 0) {
            maskedDims[maskedCnt++] = i;
        }
    }

    if (maskedCnt == 0) {
        // 退化场景：没有维度参与索引，输出等于输入
        identity_ = true;
        preCnt_ = 0;
        preProd_ = 1;
        outerCnt_ = 0;
        outerNum_ = 1;
        for (int64_t i = 0; i < ndim_ - 1; ++i) {
            outerDims_[outerCnt_++] = i;
        }
        for (int64_t i = 0; i < outerCnt_; ++i) {
            outerNum_ *= dims_[outerDims_[i]];
        }
        tail_ = dims_[ndim_ - 1];
        rr_ = outerNum_ * tail_;
        maskCnt_ = 0;
        return;
    }

    int64_t firstM = maskedDims[0];
    int64_t lastM = maskedDims[maskedCnt - 1];
    // 题面 3.1 参考实现：所有被索引维度复用同一个 indices 值
    maskCnt_ = maskedCnt;
    for (int64_t i = 0; i < maskedCnt; ++i) {
        maskDims_[i] = maskedDims[i];
    }
    tail_ = 1;
    for (int64_t i = lastM + 1; i < ndim_; ++i) {
        tail_ *= dims_[i];
    }

    bool contiguous = (lastM - firstM + 1 == maskedCnt);
    preCnt_ = 0;
    preProd_ = 1;
    outerCnt_ = 0;
    outerNum_ = 1;
    if (contiguous) {
        // 连续轴场景：前缀维度在输出中位于 indices 之前
        for (int64_t i = 0; i < firstM; ++i) {
            preDims_[preCnt_++] = i;
        }
    } else {
        // 非连续轴场景：其余所有非索引维度按原顺序位于 indices 之后
        for (int64_t i = 0; i <= lastM; ++i) {
            bool isMasked = false;
            for (int64_t j = 0; j < maskedCnt; ++j) {
                if (maskedDims[j] == i) {
                    isMasked = true;
                    break;
                }
            }
            if (!isMasked) {
                outerDims_[outerCnt_++] = i;
            }
        }
    }
    for (int64_t i = 0; i < preCnt_; ++i) {
        preProd_ *= dims_[preDims_[i]];
    }
    // 前缀维度的合并步长 = 最后一个前缀维度在 x 中的 stride（前缀混合基索引 p 的 x 偏移 = p*该 stride）
    preStride_ = (preCnt_ > 0) ? strides_[preDims_[preCnt_ - 1]] : 0;
    for (int64_t i = 0; i < outerCnt_; ++i) {
        outerNum_ *= dims_[outerDims_[i]];
    }
    rr_ = outerNum_ * tail_;
}

template <typename T>
__aicore__ inline void Index<T>::Prepare(GM_ADDR x, GM_ADDR indices, GM_ADDR y)
{
    xTotal_ = 1;
    for (int64_t i = 0; i < ndim_; ++i) {
        xTotal_ *= dims_[i];
    }
    outTotal_ = identity_ ? xTotal_ : (preProd_ * q_ * rr_);

    int64_t elemSize = static_cast<int64_t>(sizeof(T));
    tailChunk_ = tail_;
    if (tailChunk_ > TAIL_CHUNK_MAX) {
        tailChunk_ = TAIL_CHUNK_MAX;
    }
    if (tailChunk_ < 1) {
        tailChunk_ = 1;
    }
    int64_t tailBytes = ((UB_ELEMS_MAX * elemSize * byteUnits_ + 31) / 32) * 32;
    pipe.InitBuffer(bufA_, tailBytes);
    pipe.InitBuffer(bufB_, tailBytes);

    // indices 缓冲区按可能的最大分块分配，兼容 q_ 很大时分块读取
    int64_t idxBytes = ((IDX_BUF_ELEMS * 8 + 31) / 32) * 32;
    pipe.InitBuffer(idxBuf, idxBytes);

    uint64_t xSize = xTotal_ > 0 ? static_cast<uint64_t>(xTotal_ * byteUnits_) : 1ULL;
    uint64_t ySize = outTotal_ > 0 ? static_cast<uint64_t>(outTotal_ * byteUnits_) : 1ULL;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), xSize);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), ySize);
    if (q_ > 0) {
        indicesGM.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices), static_cast<uint64_t>(q_));
    }

    totalUnits_ = preProd_ * q_;

    idxCached_ = (q_ > 0) && (q_ <= IDX_BUF_ELEMS);
    if (idxCached_) {
        CopyIndicesChunk(0, q_);
        idxChunk_ = 0;
    }
}

template <typename T>
__aicore__ inline void Index<T>::CopyIndicesChunk(int64_t startElem, int64_t count)
{
    LocalTensor<int32_t> idxLocal = idxBuf.Get<int32_t>();
    GlobalTensor<int32_t> idxGM;
    idxGM.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(indicesAddr_) + startElem * 2, static_cast<uint64_t>(count * 2));
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * 8), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, static_cast<int32_t>(0)};
    DataCopyPad(idxLocal, idxGM, copyParams, padParams);
    // 同 DeriveLayout：DMA 完成后才能安全地读标量
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline int64_t Index<T>::IndexValue(int64_t t)
{
    if (idxCached_) {
        LocalTensor<int32_t> idxLocal = idxBuf.Get<int32_t>();
        return ReadInt64(idxLocal, t);
    }
    int64_t chunkIdx = t / IDX_BUF_ELEMS;
    if (chunkIdx != idxChunk_) {
        int64_t start = chunkIdx * IDX_BUF_ELEMS;
        int64_t count = q_ - start;
        if (count > IDX_BUF_ELEMS) {
            count = IDX_BUF_ELEMS;
        }
        CopyIndicesChunk(start, count);
        idxChunk_ = chunkIdx;
    }
    LocalTensor<int32_t> idxLocal = idxBuf.Get<int32_t>();
    return ReadInt64(idxLocal, t - chunkIdx * IDX_BUF_ELEMS);
}

template <typename T>
__aicore__ inline int64_t Index<T>::DecodeOffset(int64_t idx, const int64_t* dimList, int64_t cnt) const
{
    int64_t off = 0;
    int64_t tmp = idx;
    for (int64_t i = cnt - 1; i >= 0; --i) {
        int64_t d = dimList[i];
        int64_t sz = dims_[d];
        if (sz <= 0) {
            continue;
        }
        off += (tmp % sz) * strides_[d];
        tmp /= sz;
    }
    return off;
}

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Index<T>::CopyInBlocks(int64_t progress, int64_t blkCount, int64_t blkLen, int64_t srcGapBytes)
{
    int32_t b = curBuf_;
    if (pending_[b]) {
        WaitFlag<HardEvent::MTE3_MTE2>(b);
        pending_[b] = false;
    }
    LocalTensor<T> inLocal = (b == 0) ? bufA_.Get<T>() : bufB_.Get<T>();
    uint32_t lenBytes = static_cast<uint32_t>(blkLen * static_cast<int64_t>(sizeof(T)) * byteUnits_);
    DataCopyExtParams copyParams{static_cast<uint16_t>(blkCount), lenBytes,
                                 static_cast<uint32_t>(srcGapBytes), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(inLocal, inputGMX[progress * byteUnits_], copyParams, padParams);
}

template <typename T>
__aicore__ inline void Index<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    CopyInBlocks(progress, 1, currentNum, 0);
}

template <typename T>
__aicore__ inline void Index<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    (void)currentNum;
    // 纯搬运：只需等待本缓冲的 MTE2 搬入完成，随后由 MTE3 搬出（不使用向量指令，逐位精确）
    int32_t b = curBuf_;
    SetFlag<HardEvent::MTE2_MTE3>(b);
    WaitFlag<HardEvent::MTE2_MTE3>(b);
}

template <typename T>
__aicore__ inline void Index<T>::CopyOutBlocks(int64_t progress, int64_t blkCount, int64_t blkLen, int64_t dstGapBytes)
{
    int32_t b = curBuf_;
    LocalTensor<T> outLocal = (b == 0) ? bufA_.Get<T>() : bufB_.Get<T>();
    uint32_t lenBytes = static_cast<uint32_t>(blkLen * static_cast<int64_t>(sizeof(T)) * byteUnits_);
    DataCopyExtParams copyParams{static_cast<uint16_t>(blkCount), lenBytes, 0,
                                 static_cast<uint32_t>(dstGapBytes), 0};
    DataCopyPad(outputGMY[progress * byteUnits_], outLocal, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(b);
    pending_[b] = true;
    curBuf_ ^= 1;
}

template <typename T>
__aicore__ inline void Index<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    CopyOutBlocks(progress, 1, currentNum, 0);
}

// 一次 strided 块 DMA：搬 blkCount 个「blkLen 个连续元素」的块，
// 源块间隔 srcStep（元素）、目标块间隔 dstStep（元素）。
// 用 DataCopyExtParams 的 blockCount/srcStride 把「非索引维度的重复搬运」合并为少量 DMA，
// 解决「小尾块 × 大量小 DMA」的 latency 受限问题。
template <typename T>
__aicore__ inline void Index<T>::BlockCopy(int64_t src0, int64_t dst0, int64_t blkLen, int64_t srcStep,
                                           int64_t dstStep, int64_t blkCount)
{
    if (blkLen <= 0 || blkCount <= 0 || srcStep < blkLen || dstStep < blkLen) {
        return;
    }
    int64_t srcGap = (srcStep - blkLen) * static_cast<int64_t>(sizeof(T)) * byteUnits_;
    int64_t dstGap = (dstStep - blkLen) * static_cast<int64_t>(sizeof(T)) * byteUnits_;
    if (srcGap < 0 || dstGap < 0 || srcGap > 0xFFFFFFFFLL || dstGap > 0xFFFFFFFFLL) {
        // 间隔超界：退化为逐块搬运
        for (int64_t c = 0; c < blkCount; ++c) {
            Gather(src0 + c * srcStep, dst0 + c * dstStep, blkLen);
        }
        return;
    }
    int64_t per = UB_ELEMS_MAX / blkLen;
    if (per > 32767) {
        per = 32767;
    }
    if (per < 1) {
        per = 1;
    }
    int64_t done = 0;
    while (done < blkCount) {
        int64_t cnt = blkCount - done;
        if (cnt > per) {
            cnt = per;
        }
        CopyInBlocks(src0 + done * srcStep, cnt, blkLen, srcGap);
        Compute(cnt * blkLen);
        CopyOutBlocks(dst0 + done * dstStep, cnt, blkLen, dstGap);
        done += cnt;
    }
}

template <typename T>
__aicore__ inline void Index<T>::Drain()
{
    for (int32_t b = 0; b < 2; ++b) {
        if (pending_[b]) {
            WaitFlag<HardEvent::MTE3_MTE2>(b);
            pending_[b] = false;
        }
    }
}

template <typename T>
__aicore__ inline void Index<T>::Gather(int64_t srcOff, int64_t dstOff, int64_t len)
{
    // 防御：索引数据越界（题面明确"越界行为不做保证"）时，将其夹到合法区间，
    // 避免真机向量核非法地址访问（ACL_ERROR_RT_VECTOR_CORE_EXCEPTION 507035）。
    if (xTotal_ <= 0 || len <= 0 || len > xTotal_) {
        return;
    }
    int64_t src = srcOff;
    if (src < 0) {
        src = 0;
    }
    if (src > xTotal_ - len) {
        src = xTotal_ - len;
    }
    int64_t done = 0;
    while (done < len) {
        int64_t chunk = len - done;
        if (chunk > tailChunk_) {
            chunk = tailChunk_;
        }
        CopyIn(src + done, chunk);
        Compute(chunk);
        CopyOut(dstOff + done, chunk);
        done += chunk;
    }
}

template <typename T>
__aicore__ inline void Index<T>::IdentityCopy()
{
    int64_t done = 0;
    while (done < xTotal_) {
        int64_t chunk = xTotal_ - done;
        if (chunk > tailChunk_) {
            chunk = tailChunk_;
        }
        CopyIn(done, chunk);
        Compute(chunk);
        CopyOut(done, chunk);
        done += chunk;
    }
    Drain();
}

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Index<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (!active_) {
        return;
    }
    if (identity_) {
        IdentityCopy();
        return;
    }
    if (totalUnits_ <= 0 || q_ <= 0) {
        return;
    }

    int64_t blockNum = GetBlockNum();
    if (blockNum <= 0) {
        blockNum = 1;
    }
    int64_t unitsPerCore = (totalUnits_ + blockNum - 1) / blockNum;
    int64_t u0 = GetBlockIdx() * unitsPerCore;
    int64_t u1 = u0 + unitsPerCore;
    if (u1 > totalUnits_) {
        u1 = totalUnits_;
    }
    for (int64_t u = u0; u < u1; ++u) {
        int64_t p = u / q_;
        int64_t t = u - p * q_;
        int64_t idxVal = IndexValue(t);
        // 所有被索引维度复用同一 indices 值；每个维度按自身长度取模（支持负索引，
        // 与 Ascend 官方 Index 规范的 [-dim, dim-1] 取值域一致）
        int64_t idxOff = 0;
        for (int64_t j = 0; j < maskCnt_; ++j) {
            int64_t d = maskDims_[j];
            int64_t sz = dims_[d];
            if (sz <= 0) {
                continue;
            }
            int64_t c = idxVal % sz;
            if (c < 0) {
                c += sz;
            }
            idxOff += c * strides_[d];
        }
        int64_t outBase = u * rr_;
        if (outerCnt_ == 0) {
            if (preCnt_ == 0) {
                // 完全连续的单次搬运
                Gather(idxOff, outBase, tail_);
            } else {
                int64_t preOff = DecodeOffset(p, preDims_, preCnt_);
                Gather(preOff + idxOff, outBase, tail_);
            }
        } else {
            // 非连续轴：把「最内层外层维度」的重复搬运合并为一次 strided 块 DMA
            int64_t inner = outerDims_[outerCnt_ - 1];
            int64_t innerSz = dims_[inner];
            int64_t innerStride = strides_[inner];
            int64_t combos = outerNum_ / innerSz;
            if (innerSz > 1 && tail_ <= tailChunk_ && CanBatch(tail_)) {
                for (int64_t kk = 0; kk < combos; ++kk) {
                    int64_t remOff = DecodeOffset(kk, outerDims_, outerCnt_ - 1);
                    BlockCopy(idxOff + remOff, outBase + kk * innerSz * tail_, tail_, innerStride, tail_, innerSz);
                }
            } else {
                int64_t base = idxOff;
                for (int64_t k = 0; k < outerNum_; ++k) {
                    int64_t remOff = DecodeOffset(k, outerDims_, outerCnt_);
                    Gather(base + remOff, outBase + k * tail_, tail_);
                }
            }
        }
    }
    Drain();
}

} // namespace NsIndex
#endif // INDEX_H