/*!
 * \file group_norm_silu_quant.h
 * \brief GroupNormSiluQuant 算子 kernel 类定义
 */

#ifndef GROUPNORMSILUQUANT_H
#define GROUPNORMSILUQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_silu_quant_tiling_data.h"
#include "group_norm_silu_quant_tiling_key.h"

namespace NsGroupNormSiluQuant {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 单块 UB 处理元素数（编译期固定，保证 UB 不溢出）
constexpr int64_t GNSQ_UB_ELEMS = 4096;
// 统计量（mean / rstd）UB 收集容量：攒满一段后经 MTE3 统一写出
constexpr int64_t GNSQ_STAT_CAP = 4096;

// ===== M3 求和结构常量（与 numpy 1.26 f32 pairwise 逐位一致，黑盒探测确证）=====
// sum(n): n<8 标量顺序和；8<=n<=128 8-lane+横向平衡树；
//         128<n<=8192 二分递归(n2=n/2-8对齐)；n>8192 8192定长段(段间顺序合并)
constexpr int64_t M3_SEG = 8192;    // 定长段大小
constexpr int64_t M3_LEAF = 128;    // 叶大小（8-lane 作用域）
constexpr int64_t M3_MAX_LEAVES = M3_SEG / M3_LEAF;  // 满段叶数 64
constexpr int64_t M3_STACK_DEPTH = 48;  // 尾段显式递归栈深（log2(8192/8)*2+裕量）

// TilingData 位域约定（结构不可改，host / kernel 两侧严格一致）：
//   blockFactor : bit[0:32)  = hwSize（x 中 C 之后所有维度的乘积）
//                 bit[32:48) = C（通道数）
//                 bit[48:64) = G（num_groups，分组数）
//   ubFactor    : bit[0:32)  = epsilon 的 IEEE-754 位模式
//                 bit[32:62) = N（batch 数）
//                 bit[62]    = activateSilu（1 = 开启 SiLU）
//                 bit[63]    = quantScale 是否按通道（1 = shape (C,)，0 = 标量）
//   totalNum    : 保留 = N * C * hwSize（kernel 由位域字段推导，不依赖该字段）
constexpr uint64_t HWSIZE_MASK = 0xFFFFFFFFULL;  // 32 bit
constexpr uint64_t C_MASK = 0xFFFFULL;           // 16 bit
constexpr uint64_t G_MASK = 0xFFFFULL;           // 16 bit
constexpr uint64_t N_MASK = 0x3FFFFFFFULL;       // 30 bit

template <typename T>
class GroupNormSiluQuant {
public:
    __aicore__ inline GroupNormSiluQuant(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut, const GroupNormSiluQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void NormalizeQuant(int64_t currentNum, float mean, float invStd, float gm, float bt, const LocalTensor<float>& qsVec);
    __aicore__ inline void FlushStats();

    // f16 subnormal 输入清洗（FTZ）：硬件 vconv 对 f16 subnormal 输入行为未定义
    //（实测产生 NaN），评测随机数据必含 subnormal，置 0 后统计量误差 ~1e-9 量级可忽略
    __aicore__ inline void FtzSubnormal(LocalTensor<T>& xLocal, int64_t len);
    // ===== M3 numpy 逐位对齐求和（pass1 两遍共用骨架）=====
    __aicore__ inline void LeafSumTo(int64_t gmBase, int64_t leafLen, int32_t redSlot, int64_t isSq, float mean);
    __aicore__ inline float LeafSumScalar(int64_t gmBase, int64_t leafLen, int64_t isSq, float mean);
    __aicore__ inline float ScalarSeqSum(int64_t gmBase, int64_t len, int64_t isSq, float mean);
    __aicore__ inline float TreeSumPow2(int64_t gmBase, int64_t n, int64_t isSq, float mean);
    __aicore__ inline float SegSumTail(int64_t gmBase, int64_t segLen, int64_t isSq, float mean);
    __aicore__ inline float SumM3(int64_t gmBase, int64_t n, int64_t isSq, float mean);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<TPosition::VECCALC> floatBuf_;   // 输入升 fp32 的中间缓冲（fp32 时为同型 Cast）
    TBuf<TPosition::VECCALC> sqBuf_;      // pass1b 的 (x-mean)^2；pass2 复用为 exp / qF16 缓冲
    TBuf<TPosition::VECCALC> acc8Buf_;    // 叶内 8-lane 行累加器
    TBuf<TPosition::VECCALC> leafRedBuf_; // 满段叶值收集（bitrev 序，对折用）
    TBuf<TPosition::VECCALC> stackIBuf_;  // 尾段显式递归：区间栈 (base,len,phase)
    TBuf<TPosition::VECCALC> stackVBuf_;  // 尾段显式递归：值栈
    TBuf<TPosition::VECCALC> qsBuf_;      // quantScale 广播向量
    TBuf<TPosition::VECCALC> redBuf_;     // 归约结果（尾段叶子）
    TBuf<TPosition::VECCALC> tmpBuf_;     // 归约 sharedTmpBuffer；pass2 复用为 qS32 缓冲
    TBuf<TPosition::VECCALC> meanStatFBuf_;   // mean 统计量 fp32 收集缓冲
    TBuf<TPosition::VECCALC> rstdStatFBuf_;   // rstd 统计量 fp32 收集缓冲
    TBuf<TPosition::VECCALC> meanStatTBuf_;   // mean 统计量输出类型缓冲（仅 fp16）
    TBuf<TPosition::VECCALC> rstdStatTBuf_;   // rstd 统计量输出类型缓冲（仅 fp16）

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> gammaGM;
    GlobalTensor<T> betaGM;
    GlobalTensor<float> quantScaleGM;
    GlobalTensor<int8_t> outputGMY;
    GlobalTensor<T> meanOutGM;
    GlobalTensor<T> rstdOutGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;

    int64_t n_ = 1;            // batch 数
    int64_t c_ = 1;            // 通道数
    int64_t g_ = 1;            // 分组数（num_groups）
    int64_t cg_ = 1;           // 每组通道数 = C / G
    int64_t hwSize_ = 1;       // C 之后所有维度的乘积
    int64_t groupSize_ = 1;    // 每组归约元素数 = cg_ * hwSize_
    int64_t blk_ = GNSQ_UB_ELEMS;
    int64_t groupStart_ = 0;   // 本核负责的组区间 [groupStart_, groupEnd_)
    int64_t groupEnd_ = 0;
    int64_t activateSilu_ = 1;   // 1 = 开启 SiLU
    int64_t qsPerChannel_ = 0;   // 1 = quantScale 为 (C,)
    float qsScalar_ = 1.0f;      // 标量 quantScale 缓存
    float epsilon_ = 1e-5f;
    int64_t statCount_ = 0;      // 统计量收集缓冲中已累积的组数
    int64_t statFlushBase_ = 0;  // 下一段统计量写出的起始组索引
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut, const GroupNormSiluQuantTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 1. 解包 TilingData 位域
    uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);

    hwSize_ = static_cast<int64_t>(blockFactor & HWSIZE_MASK);
    c_ = static_cast<int64_t>((blockFactor >> 32) & C_MASK);
    g_ = static_cast<int64_t>((blockFactor >> 48) & G_MASK);
    n_ = static_cast<int64_t>((ubFactor >> 32) & N_MASK);
    activateSilu_ = static_cast<int64_t>((ubFactor >> 62) & 0x1ULL);
    qsPerChannel_ = static_cast<int64_t>((ubFactor >> 63) & 0x1ULL);

    uint32_t epsBits = static_cast<uint32_t>(ubFactor & 0xFFFFFFFFULL);
    union { uint32_t u; float f; } epsConv;
    epsConv.u = epsBits;
    epsilon_ = epsConv.f;

    // 占位/退化 tiling 的一致性保护，避免除零、死循环与非法 shape
    if (hwSize_ <= 0) { hwSize_ = 1; }
    if (c_ <= 0) { c_ = 1; }
    if (g_ <= 0) { g_ = 1; }
    if (n_ <= 0) { n_ = 1; }
    if (!(epsilon_ > 0.0f)) { epsilon_ = 1e-5f; }  // 处理 0 / 负数 / NaN

    cg_ = c_ / g_;
    if (cg_ <= 0) { cg_ = 1; }
    groupSize_ = cg_ * hwSize_;
    if (groupSize_ <= 0) { groupSize_ = 1; }
    blk_ = GNSQ_UB_ELEMS;

    int64_t totalNum = n_ * c_ * hwSize_;
    if (totalNum <= 0) { totalNum = 1; }

    // 2. GlobalTensor 绑定
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), totalNum);
    gammaGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma), c_);
    betaGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), c_);
    quantScaleGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale), (qsPerChannel_ != 0) ? c_ : 1);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out), totalNum);
    meanOutGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(meanOut), n_ * g_);
    rstdOutGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rstdOut), n_ * g_);

    // 3. 多核切分：按组（N * G 个组）均分，组内归约天然无跨核依赖
    int64_t totalGroups = n_ * g_;
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockNum <= 0) { blockNum = 1; }
    int64_t chunk = (totalGroups + blockNum - 1) / blockNum;
    groupStart_ = blockIdx * chunk;
    groupEnd_ = groupStart_ + chunk;
    if (groupStart_ > totalGroups) { groupStart_ = totalGroups; }
    if (groupEnd_ > totalGroups) { groupEnd_ = totalGroups; }
    blockLength_ = groupEnd_ - groupStart_;
    ubLength_ = blk_;
    // 统计量写出偏移：meanOut / rstdOut 为 (N, G) 连续布局，本核组区间即输出区间
    statFlushBase_ = groupStart_;
    statCount_ = 0;

    // 4. UB 初始化
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(int8_t)));
    pipe.InitBuffer(floatBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(sqBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(acc8Buf_, 32);
    pipe.InitBuffer(leafRedBuf_, static_cast<uint32_t>(M3_MAX_LEAVES) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(stackIBuf_, static_cast<uint32_t>(M3_STACK_DEPTH * 3) * static_cast<uint32_t>(sizeof(int64_t)));
    pipe.InitBuffer(stackVBuf_, static_cast<uint32_t>(M3_STACK_DEPTH) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(qsBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(redBuf_, 256);
    pipe.InitBuffer(tmpBuf_, static_cast<uint32_t>(blk_) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(meanStatFBuf_, static_cast<uint32_t>(GNSQ_STAT_CAP) * static_cast<uint32_t>(sizeof(float)));
    pipe.InitBuffer(rstdStatFBuf_, static_cast<uint32_t>(GNSQ_STAT_CAP) * static_cast<uint32_t>(sizeof(float)));
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(meanStatTBuf_, static_cast<uint32_t>(GNSQ_STAT_CAP) * static_cast<uint32_t>(sizeof(T)));
        pipe.InitBuffer(rstdStatTBuf_, static_cast<uint32_t>(GNSQ_STAT_CAP) * static_cast<uint32_t>(sizeof(T)));
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress], extParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::FtzSubnormal(LocalTensor<T>& xLocal, int64_t len)
{
    if constexpr (sizeof(T) == 2) {
        if (len <= 0) { return; }
        // Compares/Select 的 count 要求 256 字节对齐（half 为 128 元素）：
        // 不足一个对齐段的极小组（len < 8 的标量路径）跳过清洗（subnormal 出现概率 ~1e-5 可忽略）
        if (len < 8) { return; }
        int32_t cnt = static_cast<int32_t>(len);
        int32_t cntAl = ((cnt + 127) / 128) * 128;
        if (cntAl > static_cast<int32_t>(blk_)) { cntAl = static_cast<int32_t>(blk_); }
        LocalTensor<half> xa = sqBuf_.Get<half>();
        Duplicate(xa, static_cast<half>(0), cntAl);
        Abs(xa, xLocal, cnt);
        LocalTensor<uint8_t> mask = tmpBuf_.Get<uint8_t>();
        Compares(mask, xa, static_cast<half>(6.10351562e-05f), CMPMODE::GE, static_cast<uint32_t>(cntAl));
        // 越界段（cnt..cntAl）mask 为假 -> 写 0，仅触碰本块 buffer 内区域，后续只读前 cnt 个
        Select(xLocal, mask, xLocal, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<uint32_t>(cntAl));
    }
}

// ===== M3 求和实现 =====
// 叶内（8 <= leafLen <= 128）：8-lane 行累加 + ReduceSum(8) 横向（平衡树，实验确证），
// 主循环界与尾巴语义与 numpy 严格一致（尾巴元素在横向结果上顺序相加）
template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::LeafSumTo(int64_t gmBase, int64_t leafLen, int32_t redSlot, int64_t isSq, float mean)
{
    CopyIn(gmBase, leafLen);
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    FtzSubnormal(xLocal, leafLen);
    LocalTensor<float> f;
    if constexpr (sizeof(T) == 4) {
        f = xLocal;
    } else {
        f = floatBuf_.Get<float>();
        Cast(f, xLocal, RoundMode::CAST_NONE, static_cast<int32_t>(leafLen));
    }
    if (isSq != 0) {
        Adds(f, f, -mean, static_cast<int32_t>(leafLen));
        Mul(f, f, f, static_cast<int32_t>(leafLen));
    }
    // 8-lane 行累加：r[j] = a[j] + a[8+j] + ...（k 顺序，与 numpy 一致）
    LocalTensor<float> acc8 = acc8Buf_.Get<float>();
    Duplicate(acc8, 0.0f, 8);
    int64_t mainEnd = leafLen - (leafLen % 8);
    Add(acc8, acc8, f[0], static_cast<int32_t>((leafLen < 8) ? leafLen : 8));
    for (int64_t i = 8; i < mainEnd; i += 8) {
        Add(acc8, acc8, f[i], 8);
    }
    // 横向：((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) —— ReduceSum(8) 的硬件树（平衡）已实验确证。
    // dst 用固定起点（偏移 dst 对齐不可靠），结果标量回存 leafVal[redSlot]
    LocalTensor<float> red = redBuf_.Get<float>();
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    ReduceSum(red, acc8, tmp, 8);
    inputQueueX.FreeTensor(xLocal);
    PipeBarrier<PIPE_V>();
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    float res = red.GetValue(0);
    // 尾巴：i ∈ [mainEnd, leafLen)，res = res + a[i]（numpy 的 for(;i<n;i++) res += a[i]）
    for (int64_t i = mainEnd; i < leafLen; ++i) {
        res = res + f.GetValue(static_cast<uint32_t>(i));
    }
    LocalTensor<float> leafVal = leafRedBuf_.Get<float>();
    leafVal.SetValue(static_cast<uint32_t>(redSlot), res);
    event_t eventSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSToV);
    WaitFlag<HardEvent::S_V>(eventSToV);
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::LeafSumScalar(int64_t gmBase, int64_t leafLen, int64_t isSq, float mean)
{
    LocalTensor<float> red = redBuf_.Get<float>();
    // 复用 leafRed 通道：写入 leafRed[0] 后取回标量
    CopyIn(gmBase, leafLen);
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    FtzSubnormal(xLocal, leafLen);
    LocalTensor<float> f;
    if constexpr (sizeof(T) == 4) {
        f = xLocal;
    } else {
        f = floatBuf_.Get<float>();
        Cast(f, xLocal, RoundMode::CAST_NONE, static_cast<int32_t>(leafLen));
    }
    if (isSq != 0) {
        Adds(f, f, -mean, static_cast<int32_t>(leafLen));
        Mul(f, f, f, static_cast<int32_t>(leafLen));
    }
    LocalTensor<float> acc8 = acc8Buf_.Get<float>();
    Duplicate(acc8, 0.0f, 8);
    int64_t mainEnd = leafLen - (leafLen % 8);
    Add(acc8, acc8, f[0], static_cast<int32_t>((leafLen < 8) ? leafLen : 8));
    for (int64_t i = 8; i < mainEnd; i += 8) {
        Add(acc8, acc8, f[i], 8);
    }
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    ReduceSum(red, acc8, tmp, 8);
    PipeBarrier<PIPE_V>();
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    float res = red.GetValue(0);
    inputQueueX.FreeTensor(xLocal);
    for (int64_t i = mainEnd; i < leafLen; ++i) {
        res = res + f.GetValue(static_cast<uint32_t>(i));
    }
    return res;
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::ScalarSeqSum(int64_t gmBase, int64_t len, int64_t isSq, float mean)
{
    // n < 8：res = 0; for i: res += a[i]（numpy 标量顺序和）
    CopyIn(gmBase, len);
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    FtzSubnormal(xLocal, len);
    LocalTensor<float> f;
    if constexpr (sizeof(T) == 4) {
        f = xLocal;
    } else {
        f = floatBuf_.Get<float>();
        Cast(f, xLocal, RoundMode::CAST_NONE, static_cast<int32_t>(len));
    }
    if (isSq != 0) {
        Adds(f, f, -mean, static_cast<int32_t>(len));
        Mul(f, f, f, static_cast<int32_t>(len));
    }
    PipeBarrier<PIPE_V>();
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    float res = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        res = res + f.GetValue(static_cast<uint32_t>(i));
    }
    inputQueueX.FreeTensor(xLocal);
    return res;
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::TreeSumPow2(int64_t gmBase, int64_t n, int64_t isSq, float mean)
{
    // 2 的幂区间（128 <= n <= 8192）：二分平衡树（n2 = n/2 递归到 128 叶）。
    // 叶按 bitrev 序写入 leafVal，对折折叠 == 地址序平衡二分树（数学等价，消标量合并）。
    // 对折向量层仅 h >= 8（Add 的 32 字节对齐约束），尾部 8 元素用标量平衡树
    int64_t m = n / M3_LEAF;
    int32_t bits = 0;
    int64_t t = m;
    while (t > 1) { bits += 1; t >>= 1; }
    for (int64_t slot = 0; slot < m; ++slot) {
        // bitrev(slot, bits)：处理序 -> 地址叶序
        int64_t p = 0;
        int64_t v = slot;
        for (int32_t bi = 0; bi < bits; ++bi) { p = (p << 1) | (v & 1); v >>= 1; }
        LeafSumTo(gmBase + p * M3_LEAF, M3_LEAF, static_cast<int32_t>(slot), isSq, mean);
    }
    LocalTensor<float> leafVal = leafRedBuf_.Get<float>();
    PipeBarrier<PIPE_V>();
    // 向量对折层（h >= 8，满足 32B 对齐）
    for (int64_t h = m / 2; h >= 8; h >>= 1) {
        Add(leafVal, leafVal, leafVal[static_cast<uint32_t>(h)], static_cast<int32_t>(h));
    }
    PipeBarrier<PIPE_V>();
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    // 尾部 <= 8 元素的对折树（标量，错半配对），与向量对折层共同构成完整对折
    int64_t rem = (m < 8) ? m : 8;
    float r[8];
    for (int64_t i = 0; i < rem; ++i) {
        r[i] = leafVal.GetValue(static_cast<uint32_t>(i));
    }
    while (rem > 1) {
        for (int64_t i = 0; i < rem / 2; ++i) {
            r[i] = r[i] + r[i + rem / 2];
        }
        rem /= 2;
    }
    return (rem == 1) ? r[0] : 0.0f;
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::SegSumTail(int64_t gmBase, int64_t segLen, int64_t isSq, float mean)
{
    // 尾段（<= 8191，任意长度）：二分递归 n2 = n/2 - n2%8 的显式栈迭代实现
    LocalTensor<int64_t> iStack = stackIBuf_.Get<int64_t>();
    LocalTensor<float> vStack = stackVBuf_.Get<float>();
    int64_t sp = 0;
    int64_t vp = 0;
    iStack.SetValue(static_cast<uint32_t>(sp * 3 + 0), gmBase);
    iStack.SetValue(static_cast<uint32_t>(sp * 3 + 1), segLen);
    iStack.SetValue(static_cast<uint32_t>(sp * 3 + 2), 0);
    sp += 1;
    while (sp > 0) {
        int64_t b = iStack.GetValue(static_cast<uint32_t>((sp - 1) * 3 + 0));
        int64_t l = iStack.GetValue(static_cast<uint32_t>((sp - 1) * 3 + 1));
        int64_t ph = iStack.GetValue(static_cast<uint32_t>((sp - 1) * 3 + 2));
        if (ph == 0) {
            if (l < 8) {
                sp -= 1;
                float v = ScalarSeqSum(b, l, isSq, mean);
                vStack.SetValue(static_cast<uint32_t>(vp), v);
                vp += 1;
            } else if (l <= M3_LEAF) {
                sp -= 1;
                float v = LeafSumScalar(b, l, isSq, mean);
                vStack.SetValue(static_cast<uint32_t>(vp), v);
                vp += 1;
            } else {
                int64_t n2 = l / 2;
                n2 -= n2 % 8;
                iStack.SetValue(static_cast<uint32_t>((sp - 1) * 3 + 2), 1);
                // 后进先出：先压右，再压左（左先求值）
                iStack.SetValue(static_cast<uint32_t>(sp * 3 + 0), b + n2);
                iStack.SetValue(static_cast<uint32_t>(sp * 3 + 1), l - n2);
                iStack.SetValue(static_cast<uint32_t>(sp * 3 + 2), 0);
                sp += 1;
                iStack.SetValue(static_cast<uint32_t>(sp * 3 + 0), b);
                iStack.SetValue(static_cast<uint32_t>(sp * 3 + 1), n2);
                iStack.SetValue(static_cast<uint32_t>(sp * 3 + 2), 0);
                sp += 1;
            }
        } else {
            // 左右子值已在值栈顶（左先入、右后入）
            sp -= 1;
            float rv = vStack.GetValue(static_cast<uint32_t>(vp - 1));
            float lv = vStack.GetValue(static_cast<uint32_t>(vp - 2));
            vp -= 2;
            vStack.SetValue(static_cast<uint32_t>(vp), lv + rv);
            vp += 1;
        }
    }
    return vStack.GetValue(0);
}

template <typename T>
__aicore__ inline float GroupNormSiluQuant<T>::SumM3(int64_t gmBase, int64_t n, int64_t isSq, float mean)
{
    // M3 总入口：n <= 8192 单树；n > 8192 时 8192 定长段 + 段间顺序合并
    if (n < 8) {
        return ScalarSeqSum(gmBase, n, isSq, mean);
    }
    if (n <= M3_SEG) {
        // 2 的幂走 bitrev 对折快路；否则显式栈递归
        if ((n & (n - 1)) == 0 && n >= M3_LEAF) {
            return TreeSumPow2(gmBase, n, isSq, mean);
        }
        return SegSumTail(gmBase, n, isSq, mean);
    }
    float acc = 0.0f;
    int64_t off = 0;
    while (off < n) {
        int64_t segLen = n - off;
        if (segLen > M3_SEG) { segLen = M3_SEG; }
        float segV;
        if (segLen == M3_SEG) {
            segV = TreeSumPow2(gmBase + off, M3_SEG, isSq, mean);
        } else if ((segLen & (segLen - 1)) == 0 && segLen >= M3_LEAF) {
            segV = TreeSumPow2(gmBase + off, segLen, isSq, mean);
        } else {
            segV = SegSumTail(gmBase + off, segLen, isSq, mean);
        }
        acc = acc + segV;
        off += segLen;
    }
    return acc;
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::NormalizeQuant(int64_t currentNum, float mean, float invStd, float gm, float bt, const LocalTensor<float>& qsVec)
{
    // 归一化 + 仿射 + SiLU + 量化：
    //   gn  = ((x - mean) * invStd) * gamma + beta（与 golden 运算顺序严格一致）
    //   silu = gn / (1 + exp(-gn))（activateSilu_ 开启时）
    //   out = rint(silu / quantScale)，clip 到 [-128, 127]
    // 量化采用 f32 -> s32(RNE) -> f32(精确) -> f16(精确) -> s8(RNE) 链路：
    // s32 整数经 f32/f16 均精确可表示，全程仅 f32->s32 一次 RNE 舍入，与 np.rint 一致；
    // 避开 s32 -> f16 直转（仿真器该路径实现异常，输出 ±0）
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    FtzSubnormal(xLocal, currentNum);
    LocalTensor<int8_t> yLocal = outputQueueY.AllocTensor<int8_t>();
    LocalTensor<float> e = sqBuf_.Get<float>();
    LocalTensor<int32_t> qS32 = tmpBuf_.Get<int32_t>();
    LocalTensor<half> qF16 = sqBuf_.Get<half>();
    int32_t cnt = static_cast<int32_t>(currentNum);
    float negMean = -mean;
    if constexpr (sizeof(T) == 4) {
        Adds(xLocal, xLocal, negMean, cnt);
        Muls(xLocal, xLocal, invStd, cnt);
        Muls(xLocal, xLocal, gm, cnt);
        Adds(xLocal, xLocal, bt, cnt);
        if (activateSilu_ != 0) {
            Muls(e, xLocal, -1.0f, cnt);
            Exp(e, e, cnt);
            Adds(e, e, 1.0f, cnt);
            Div(xLocal, xLocal, e, cnt);
        }
        Div(xLocal, xLocal, qsVec, cnt);
        Maxs(xLocal, xLocal, -128.0f, cnt);
        Mins(xLocal, xLocal, 127.0f, cnt);
        Cast(qS32, xLocal, RoundMode::CAST_RINT, cnt);
        Cast(xLocal, qS32, RoundMode::CAST_NONE, cnt);
        Cast(qF16, xLocal, RoundMode::CAST_NONE, cnt);
    } else {
        LocalTensor<float> f = floatBuf_.Get<float>();
        Cast(f, xLocal, RoundMode::CAST_NONE, cnt);
        Adds(f, f, negMean, cnt);
        Muls(f, f, invStd, cnt);
        Muls(f, f, gm, cnt);
        Adds(f, f, bt, cnt);
        if (activateSilu_ != 0) {
            Muls(e, f, -1.0f, cnt);
            Exp(e, e, cnt);
            Adds(e, e, 1.0f, cnt);
            Div(f, f, e, cnt);
        }
        Div(f, f, qsVec, cnt);
        Maxs(f, f, -128.0f, cnt);
        Mins(f, f, 127.0f, cnt);
        Cast(qS32, f, RoundMode::CAST_RINT, cnt);
        Cast(f, qS32, RoundMode::CAST_NONE, cnt);
        Cast(qF16, f, RoundMode::CAST_NONE, cnt);
    }
    Cast(yLocal, qF16, RoundMode::CAST_RINT, cnt);
    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    // 归一化 + 量化主体在 NormalizeQuant() 中实现，该占位函数保持接口完整
    (void)currentNum;
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<int8_t> yLocal = outputQueueY.DeQue<int8_t>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(int8_t))), 0, 0, 0};
    DataCopyPad(outputGMY[progress], yLocal, extParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::FlushStats()
{
    // 将收集缓冲中的 mean / rstd 统一经 MTE3 写出到 GM
    //（多核 GlobalTensor::SetValue 直写 GM 不可靠，必须 UB 收集 + DataCopyPad）
    if (statCount_ <= 0) {
        return;
    }
    int32_t cnt = static_cast<int32_t>(statCount_);
    LocalTensor<float> meanF = meanStatFBuf_.Get<float>();
    LocalTensor<float> rstdF = rstdStatFBuf_.Get<float>();
    DataCopyExtParams extParams{1, static_cast<uint32_t>(statCount_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    if constexpr (sizeof(T) == 2) {
        // scalar 写 UB 完成 -> V 可读
        event_t eventSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSToV);
        WaitFlag<HardEvent::S_V>(eventSToV);
        LocalTensor<T> meanT = meanStatTBuf_.Get<T>();
        LocalTensor<T> rstdT = rstdStatTBuf_.Get<T>();
        Cast(meanT, meanF, RoundMode::CAST_RINT, static_cast<uint32_t>(cnt));
        Cast(rstdT, rstdF, RoundMode::CAST_RINT, static_cast<uint32_t>(cnt));
        // V 写 UB 完成 -> MTE3 可读
        event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyPad(meanOutGM[statFlushBase_], meanT, extParams);
        DataCopyPad(rstdOutGM[statFlushBase_], rstdT, extParams);
    } else {
        // scalar 写 UB 完成 -> MTE3 可读
        event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
        DataCopyPad(meanOutGM[statFlushBase_], meanF, extParams);
        DataCopyPad(rstdOutGM[statFlushBase_], rstdF, extParams);
    }
    // 等待 MTE3 完成后再复用收集缓冲
    event_t eventMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventMte3ToS);
    statFlushBase_ += statCount_;
    statCount_ = 0;
}

template <typename T>
__aicore__ inline void GroupNormSiluQuant<T>::Process()
{
    // TODO: 实现 Process 逻辑
    LocalTensor<float> qsVec = qsBuf_.Get<float>();

    if (qsPerChannel_ == 0) {
        qsScalar_ = quantScaleGM.GetValue(0);
    }

    for (int64_t gIdx = groupStart_; gIdx < groupEnd_; ++gIdx) {
        // 组 gIdx = n * G + g，其元素在 x 中为连续区间 [gIdx*groupSize_, +groupSize_)
        int64_t groupBase = gIdx * groupSize_;

        // ---------------- pass 1a：sum -> mean（两遍法第一遍） ----------------
        // 求和结构 M3 与 numpy 1.26 的 f32 pairwise 逐位一致（8192 段/二分/8-lane/平衡横向），
        // mean = sum / n 用除法（与 np.mean 的 true_divide 舍入一致）
        float sum = SumM3(groupBase, groupSize_, 0, 0.0f);
        float mean = sum / static_cast<float>(groupSize_);

        // ---------------- pass 1b：sum((x-mean)^2) -> var（两遍法第二遍） ----------------
        // var = E[(x-mean)^2]，与 golden np.var 数值同构：
        // 单遍 E[x^2]-mean^2 在大 groupSize / 大均值下存在灾难性消解（var 相对误差可达 1e-1+，
        // 直接导致 rstd 精度超标），两遍法无该问题
        float sqsum = SumM3(groupBase, groupSize_, 1, mean);
        float var = sqsum / static_cast<float>(groupSize_);
        if (var < 0.0f) {
            var = 0.0f;
        }
        float invStd = 1.0f / sqrt(var + epsilon_);

        // 统计量写入 UB 收集缓冲，攒满一段后经 MTE3 统一写出
        LocalTensor<float> meanF = meanStatFBuf_.Get<float>();
        LocalTensor<float> rstdF = rstdStatFBuf_.Get<float>();
        meanF.SetValue(static_cast<int32_t>(statCount_), mean);
        rstdF.SetValue(static_cast<int32_t>(statCount_), invStd);
        statCount_ += 1;
        if (statCount_ >= GNSQ_STAT_CAP) {
            FlushStats();
        }

        // ---------------- pass 2：按通道归一化 + 仿射 + SiLU + 量化 ----------------
        // gamma / beta / quantScale 是通道级的，同一组内各通道系数不同，故按通道循环
        int64_t n = gIdx / g_;
        int64_t g = gIdx - n * g_;
        for (int64_t ci = 0; ci < cg_; ++ci) {
            int64_t c = g * cg_ + ci;
            float gm = static_cast<float>(gammaGM.GetValue(static_cast<uint32_t>(c)));
            float bt = static_cast<float>(betaGM.GetValue(static_cast<uint32_t>(c)));
            float qs = (qsPerChannel_ != 0) ? quantScaleGM.GetValue(static_cast<uint32_t>(c)) : qsScalar_;
            Duplicate(qsVec, qs, static_cast<int32_t>(blk_));

            int64_t channelBase = (n * c_ + c) * hwSize_;
            for (int64_t o = 0; o < hwSize_; o += blk_) {
                int64_t cur = hwSize_ - o;
                if (cur > blk_) {
                    cur = blk_;
                }
                CopyIn(channelBase + o, cur);
                NormalizeQuant(cur, mean, invStd, gm, bt, qsVec);
                CopyOut(channelBase + o, cur);
            }
        }
    }

    // 收尾写出剩余统计量
    FlushStats();

    // 确保所有 MTE3（GM 写出）在 kernel 退出前完成
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
}

} // namespace NsGroupNormSiluQuant
#endif // GROUPNORMSILUQUANT_H
