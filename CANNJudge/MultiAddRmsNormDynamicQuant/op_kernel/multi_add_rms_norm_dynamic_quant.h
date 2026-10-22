/*!
 * \file multi_add_rms_norm_dynamic_quant.h
 * \brief MultiAddRmsNormDynamicQuant 算子 kernel 类定义
 */

#ifndef MULTIADDRMSNORMDYNAMICQUANT_H
#define MULTIADDRMSNORMDYNAMICQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "multi_add_rms_norm_dynamic_quant_tiling_data.h"
#include "multi_add_rms_norm_dynamic_quant_tiling_key.h"

namespace NsMultiAddRmsNormDynamicQuant {

using namespace AscendC;

constexpr int32_t K_BLOCK_F = 1024;
// scale 攒批行组（大 B 路径）：每组连续 ROW_FACTOR 行，scale 经 SetValue 写 UB 段、
// 组尾一次 PIPE_ALL + 批量 MTE3 写 GM，消除逐行 2 次单元素 MTE3 + 6 个事件对。
// 小 B 路径（行组数 < 4*核数）保持逐行写，保核间负载均衡。
constexpr int32_t ROW_FACTOR = 128;

// 与生产 single_row_kernel 的 SetWaitFlag 惯用法等价（conv2d/softmax 生产实现同模式）：
// FetchEventID + SetFlag + WaitFlag 事件对。PIPE_ALL 会让全部 pipe 互相等待（杀流水），
// 事件对只同步特定方向：MTE2_V=等 MTE2 拷完 V 可读；V_MTE2=等 V 读完 MTE2 可覆写；
// V_MTE3=等 V 写完 MTE3 可读；MTE3_V=等 MTE3 读完 V 可覆写。
template <HardEvent E>
__aicore__ inline void SyncPipe()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(e);
    WaitFlag<E>(e);
}

// 生产 helper.h::RoundFloat2Int8 模式：fp32 原地 ReinterpretCast 链，
// src 半区依次复用为 int32/half（元素级前向读写安全，生产真机在用）。
// 输入 |值|≤127（scale=max/127 保证），无 clamp 必要。
__aicore__ inline void Round2Int8(const LocalTensor<float>& src, const LocalTensor<int8_t>& dst,
                                  int32_t cnt, const LocalTensor<float>& w)
{
    // 910B 的 Cast<float,int32> CAST_RINT 为 half-away-from-zero（sim 为 RNE）。
    // 手动 RNE（9 条向量指令/块）：r=RINT(x); d=x-float(r); sgn=RINT(d) 属于 {-1,0,1}
    // （仅 |d|==0.5 时非 0，符号即修正方向）；odd=r&1；adj=sgn*odd（sgn*|sgn|≡sgn，
    // tie 因子冗余）；r=trunc(float(r)+adj)。分块 128。
    // w 段布局（RB=128）：[0]rI [128]rf [256]sgnI [384]sgn [512]oddI
    //   [640]oddf [768]adj [896]c1I [1024]srch(half)；共 1088 float。
    const int32_t RB = 128;
    for (int32_t s = 0; s < cnt; s += RB) {
        const int32_t sc = (cnt - s) < RB ? (cnt - s) : RB;
        LocalTensor<int32_t> rI = w[0].ReinterpretCast<int32_t>();
        LocalTensor<float> rf = w[RB];
        LocalTensor<int32_t> sgnI = w[2 * RB].ReinterpretCast<int32_t>();
        LocalTensor<float> sgn = w[3 * RB];
        LocalTensor<int32_t> oddI = w[4 * RB].ReinterpretCast<int32_t>();
        LocalTensor<float> oddf = w[5 * RB];
        LocalTensor<float> adj = w[6 * RB];
        LocalTensor<int32_t> c1I = w[7 * RB].ReinterpretCast<int32_t>();
        LocalTensor<half> srch = w[8 * RB].ReinterpretCast<half>();
        Duplicate<int32_t>(c1I, 1, sc);
        Cast<int32_t, float>(rI, src[s], RoundMode::CAST_RINT, sc);   // r = away(x)
        Cast<float, int32_t>(rf, rI, RoundMode::CAST_NONE, sc);
        Sub(sgn, src[s], rf, sc);                                     // d
        Cast<int32_t, float>(sgnI, sgn, RoundMode::CAST_RINT, sc);    // sgnI 属于 {0,±1}
        Cast<float, int32_t>(sgn, sgnI, RoundMode::CAST_NONE, sc);
        And(oddI, rI, c1I, sc);                                       // r&1
        Cast<float, int32_t>(oddf, oddI, RoundMode::CAST_NONE, sc);
        Mul(adj, sgn, oddf, sc);                                      // adj = sgn*odd
        Add(rf, rf, adj, sc);                                         // r + adj
        Cast<int32_t, float>(rI, rf, RoundMode::CAST_TRUNC, sc);      // final r
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast<half, int32_t>(srch, rI, RoundMode::CAST_NONE, sc);
        Cast<int8_t, half>(dst[s], srch, RoundMode::CAST_TRUNC, sc);
    }
}




// IEEE 正确舍入的向量除法：默认 Div 用 DivAlgo::INTRINSIC（~1ulp 误差），
// 量化 q/scale 在 rint halfway 附近会翻 1 LSB → int8 bitwise FAIL（case12-15
// 自六版起 WA 的根因）。PRECISION_0ULP_FTZ_FALSE 保证 0 ulp，与 torch 真除法一致。
// sim 路径无 config 参数（__NPU_ARCH__ != 3510），用宿主 IEEE 除法同样正确。
__aicore__ inline void IeeeDiv(const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1, int32_t cnt)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    constexpr DivConfig ieeeDivCfg = {DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    Div<float, ieeeDivCfg>(dst, src0, src1, cnt);
#else
    Div<float>(dst, src0, src1, cnt);
#endif
}

// 精确复刻 torch CPU sum 的级联结构（SumKernel.cpp::vectorized_inner_sum，
// aarch64 实测 ilp_factor=4、tail_ilp_factor=1、vec width=4）：
//  - level0：acc[0][0..16) 按 16 元素连续块顺序累加
//  - 4 层级联提升：每 16 个 i 步（=256 元素）acc1+=acc0；i%4096==0 时 acc2+=acc1；
//    i%65536==0 时 acc3+=acc2
//  - 尾部向量 + 标量尾 + 合并均按 torch 顺序在标量域完成
// 只有整链 bitwise 对齐 torch，才能消除 int8 在 rint 边界的 1ulp 翻转。
__aicore__ inline int32_t TorchCeilLog2(int32_t n)
{
    if (n <= 1) {
        return 0;
    }
    int32_t m = n - 1;
    int32_t r = 0;
    while (m > 0) {
        m >>= 1;
        ++r;
    }
    return r;
}

// 逐 16 元素块把本 tile 的 x2 累加进 acc（acc 段布局：acc0=[0,16)、acc1=[16,32)、
// acc2=[32,48)、acc3=[48,64)），iState 为全局块计数（累加后自增）。
__aicore__ inline void TorchCascadeAdd(const LocalTensor<float>& acc, const LocalTensor<float>& x2,
                                       int32_t nBlocks, int32_t& iState)
{
    for (int32_t b = 0; b < nBlocks; ++b) {
        Add(acc[0], acc[0], x2[b * 16], 16);
        ++iState;
        if ((iState & 15) == 0) {
            Add(acc[16], acc[16], acc[0], 16);
                Duplicate<float>(acc[0], 0.0f, 16);
                if ((iState & 240) == 0) {
                Add(acc[32], acc[32], acc[16], 16);
                        Duplicate<float>(acc[16], 0.0f, 16);
                        if ((iState & 3840) == 0) {
                    Add(acc[48], acc[48], acc[32], 16);
                                Duplicate<float>(acc[32], 0.0f, 16);
                            }
            }
        }
    }
}

// 合并 4 层 + 尾部向量/标量尾（严格按 torch 顺序，标量域），返回整行 sum。
// tailBuf 存全局 [sizeIlp*16, H) 的 x2 值（数量 H%16 < 16）。
__aicore__ inline float TorchCascadeFinal(const LocalTensor<float>& acc, const LocalTensor<float>& tailBuf,
                                          int32_t H)
{
    Add(acc[0], acc[0], acc[16], 16);
    Add(acc[0], acc[0], acc[32], 16);
    Add(acc[0], acc[0], acc[48], 16);
    PipeBarrier<PIPE_V>();
    event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evVS);
    WaitFlag<HardEvent::V_S>(evVS);
    const int32_t vecSize = H / 4;
    const int32_t sizeIlp = H / 16;
    float a[16];
    for (int32_t l = 0; l < 16; ++l) {
        a[l] = acc.GetValue(l);
    }
    float ps0[4];
    for (int32_t l = 0; l < 4; ++l) {
        ps0[l] = a[l];
    }
    const int32_t tailBase = sizeIlp * 16;
    for (int32_t iv = sizeIlp * 4; iv < vecSize; ++iv) {
        for (int32_t l = 0; l < 4; ++l) {
            ps0[l] = ps0[l] + tailBuf.GetValue(iv * 4 + l - tailBase);
        }
    }
    for (int32_t l = 0; l < 4; ++l) {
        ps0[l] = ps0[l] + a[4 + l];
        ps0[l] = ps0[l] + a[8 + l];
        ps0[l] = ps0[l] + a[12 + l];
    }
    float fa = 0.0f;
    for (int32_t k = vecSize * 4; k < H; ++k) {
        fa = fa + tailBuf.GetValue(k - tailBase);
    }
    for (int32_t l = 0; l < 4; ++l) {
        fa = fa + ps0[l];
    }
    return fa;
}

// 正确舍入的向量除法：目的与 torch 的 IEEE 真除法一致。910B 的 vdiv(vmadd 系)为
// 近似（且无 FMA），用 Dekker TwoProd（12-bit split，乘积精确）+ Newton 精化。
// w 为 8*cnt 工作段（与 dst/a/b 互不重叠）。
__aicore__ inline void TorchDiv(const LocalTensor<float>& dst, const LocalTensor<float>& a,
                                const LocalTensor<float>& b, int32_t cnt, const LocalTensor<float>& w)
{
    // 工作段分段：段步长取 8 的倍数（各段起始须 32B 对齐），需 7*st 空间
    const int32_t st = ((cnt + 7) / 8) * 8;
    LocalTensor<float> s0 = w[0];
    LocalTensor<float> s1 = w[st];
    LocalTensor<float> s2 = w[2 * st];
    LocalTensor<float> s3 = w[3 * st];
    LocalTensor<float> s4 = w[4 * st];
    LocalTensor<float> s5 = w[5 * st];
    LocalTensor<float> s6 = w[6 * st];
    IeeeDiv(dst, a, b, cnt);                    // q0
    Mul(s0, b, dst, cnt);                       // p = b*q0
    Muls(s1, dst, 4097.0f, cnt);
    Sub(s2, s1, dst, cnt);
    Sub(s3, s1, s2, cnt);                       // qh
    Sub(s4, dst, s3, cnt);                      // ql
    Muls(s1, b, 4097.0f, cnt);
    Sub(s2, s1, b, cnt);
    Sub(s5, s1, s2, cnt);                       // bh
    Sub(s6, b, s5, cnt);                        // bl
    Mul(s1, s3, s5, cnt);
    Sub(s1, s1, s0, cnt);
    Mul(s2, s3, s6, cnt);
    Add(s1, s1, s2, cnt);
    Mul(s2, s4, s5, cnt);
    Add(s1, s1, s2, cnt);
    Mul(s2, s4, s6, cnt);
    Add(s1, s1, s2, cnt);                       // err
    Sub(s2, a, s0, cnt);
    Sub(s2, s2, s1, cnt);                       // e = a - b*q0
    IeeeDiv(s3, s2, b, cnt);                    // corr
    Add(dst, dst, s3, cnt);                     // q = q0 + corr
}

// TorchDiv 的常量除数版：b 的 Dekker split（bh/bl）已预先计算（行级标量），
// 块内省 4 条 split 指令。wbh/wbl 为 bh/bl 的广播段（cnt 元素）。
__aicore__ inline void TorchDivC(const LocalTensor<float>& dst, const LocalTensor<float>& a,
                                 const LocalTensor<float>& b, const LocalTensor<float>& wbh,
                                 const LocalTensor<float>& wbl, int32_t cnt, const LocalTensor<float>& w)
{
    const int32_t st = ((cnt + 7) / 8) * 8;
    LocalTensor<float> s0 = w[0];
    LocalTensor<float> s1 = w[st];
    LocalTensor<float> s2 = w[2 * st];
    LocalTensor<float> s3 = w[3 * st];
    LocalTensor<float> s4 = w[4 * st];
    IeeeDiv(dst, a, b, cnt);                    // q0
    Mul(s0, b, dst, cnt);                       // p = b*q0
    Muls(s1, dst, 4097.0f, cnt);
    Sub(s2, s1, dst, cnt);
    Sub(s3, s1, s2, cnt);                       // qh
    Sub(s4, dst, s3, cnt);                      // ql
    Mul(s1, s3, wbh, cnt);
    Sub(s1, s1, s0, cnt);
    Mul(s2, s3, wbl, cnt);
    Add(s1, s1, s2, cnt);
    Mul(s2, s4, wbh, cnt);
    Add(s1, s1, s2, cnt);
    Mul(s2, s4, wbl, cnt);
    Add(s1, s1, s2, cnt);                       // err
    Sub(s2, a, s0, cnt);
    Sub(s2, s2, s1, cnt);                       // e = a - b*q0
    IeeeDiv(s3, s2, b, cnt);                    // corr
    Add(dst, dst, s3, cnt);                     // q = q0 + corr
}

// IEEE 正确舍入的向量开方：默认 Sqrt 用 SqrtAlgo::INTRINSIC（~1ulp 误差）。
__aicore__ inline void IeeeSqrt(const LocalTensor<float>& dst, const LocalTensor<float>& src, int32_t cnt)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    constexpr SqrtConfig ieeeSqrtCfg = {SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
    Sqrt<float, ieeeSqrtCfg>(dst, src, cnt);
#else
    Sqrt<float>(dst, src, cnt);
#endif
}

template <typename T>
class MultiAddRmsNormDynamicQuant {
public:
    __aicore__ inline MultiAddRmsNormDynamicQuant(){};

    __aicore__ inline void Init(GM_ADDR x1a, GM_ADDR x1b, GM_ADDR x1c, GM_ADDR x1d, GM_ADDR x1e, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smoothScale1, GM_ADDR smoothScale2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2, const MultiAddRmsNormDynamicQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename InT> __aicore__ inline void Run();
    template <typename InT> __aicore__ inline void ProcessRow(int64_t row, int32_t scaleIdx = -1, LocalTensor<float> scaleOut = LocalTensor<float>(), LocalTensor<InT> batchT = LocalTensor<InT>(), LocalTensor<int8_t> batchI8 = LocalTensor<int8_t>(), int64_t nextRow = -1);
    template <typename InT>
    __aicore__ inline void LoadRowAsync(LocalTensor<InT> dst, GM_ADDR base, int64_t row, int32_t cnt);
    template <typename InT>
    __aicore__ inline void Gm2Ub(LocalTensor<InT> dst, const GlobalTensor<InT>& src, int64_t off, int32_t cnt);
    template <typename InT>
    __aicore__ inline void Gm2UbAsync(LocalTensor<InT> dst, const GlobalTensor<InT>& src, int64_t off, int32_t cnt);
    template <typename InT>
    __aicore__ inline void Ub2Gm(GlobalTensor<InT>& dst, int64_t off, const LocalTensor<InT>& src, int32_t cnt);
    template <typename InT>
    __aicore__ inline void Ub2GmAsync(GlobalTensor<InT>& dst, int64_t off, const LocalTensor<InT>& src, int32_t cnt);
    template <typename InT>
    __aicore__ inline void LoadCastF(LocalTensor<float> dst, const LocalTensor<InT>& inB, const GlobalTensor<InT>& src, int64_t off, int32_t cnt);
    __aicore__ inline float ReduceSumScalar(const LocalTensor<float>& src, int32_t count);
    __aicore__ inline float ReduceMaxScalar(const LocalTensor<float>& src, int32_t count);

private:
    int64_t bNum_ = 0;
    int32_t hNum_ = 0;
    bool isBf16_ = false;
    float eps_ = 0.0f;

    GM_ADDR gmX1a_ = nullptr;
    GM_ADDR gmX1b_ = nullptr;
    GM_ADDR gmX1c_ = nullptr;
    GM_ADDR gmX1d_ = nullptr;
    GM_ADDR gmX1e_ = nullptr;
    GM_ADDR gmX2_ = nullptr;
    GM_ADDR gmGamma_ = nullptr;
    GM_ADDR gmSm1_ = nullptr;
    GM_ADDR gmSm2_ = nullptr;
    GM_ADDR gmY1_ = nullptr;
    GM_ADDR gmY2_ = nullptr;
    GM_ADDR gmX_ = nullptr;
    GM_ADDR gmY_ = nullptr;
    GM_ADDR gmScale1_ = nullptr;
    GM_ADDR gmScale2_ = nullptr;

    TPipe pipe_;
    TBuf<TPosition::VECCALC> bufXRow_;
    TBuf<TPosition::VECCALC> bufIn_;
    TBuf<TPosition::VECCALC> bufGs_;
    TBuf<TPosition::VECCALC> bufCalc_;
    TBuf<TPosition::VECCALC> bufOut_;
    TBuf<TPosition::VECCALC> bufI8_;
    TBuf<TPosition::VECCALC> bufSmH_;
    TBuf<TPosition::VECCALC> bufRed_;
    TBuf<TPosition::VECCALC> bufAcc_;
    TBuf<TPosition::VECCALC> bufBatchT_;
    TBuf<TPosition::VECCALC> bufBatchI8_;
    bool wResident_ = false;
    bool smHalfRes_ = false;
    int32_t smSeg_ = 0;
    int32_t batchRows_ = 0;
};

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Init(GM_ADDR x1a, GM_ADDR x1b, GM_ADDR x1c, GM_ADDR x1d, GM_ADDR x1e, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smoothScale1, GM_ADDR smoothScale2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2, const MultiAddRmsNormDynamicQuantTilingData* tilingData)
{
    gmX1a_ = x1a;
    gmX1b_ = x1b;
    gmX1c_ = x1c;
    gmX1d_ = x1d;
    gmX1e_ = x1e;
    gmX2_ = x2;
    gmGamma_ = gamma;
    gmSm1_ = smoothScale1;
    gmSm2_ = smoothScale2;
    gmY1_ = y1;
    gmY2_ = y2;
    gmX_ = x;
    gmY_ = y;
    gmScale1_ = scale1;
    gmScale2_ = scale2;

    const int64_t rawUb = tilingData->ubFactor;
    hNum_ = static_cast<int32_t>(rawUb & 0x7FFF);
    isBf16_ = ((rawUb >> 15) & 1) != 0;
    const int64_t totalNum = tilingData->totalNum;
    bNum_ = (hNum_ > 0) ? (totalNum / hNum_) : 0;

    union EpsBits {
        uint32_t u;
        float f;
    };
    EpsBits epsBits;
    epsBits.u = static_cast<uint32_t>(static_cast<uint64_t>(tilingData->blockFactor) & 0xFFFFFFFFULL);
    float epsVal = epsBits.f;
    if (!(epsVal > 0.0f) || epsVal > 1e30f) {
        epsVal = 0.0f;
    }
    eps_ = epsVal;
}

template <typename T>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Process()
{
    if (bNum_ <= 0 || hNum_ <= 0) {
        return;
    }
    if (isBf16_) {
        Run<bfloat16_t>();
    } else if (Std::is_same<T, float>::value) {
        Run<float>();
    } else {
        Run<half>();
    }
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Run()
{
    const uint32_t rowBytes = static_cast<uint32_t>(hNum_) * 4u;
    pipe_.InitBuffer(bufXRow_, rowBytes + 32u); // +32B: DataCopyPad 读源按 32B 块取整的余量
    // inB 6 块（行间预取）：块0=x1a、块1=x2、块2~5=x1b~x1e，全部路一次 MTE2 发射，
    // 下一行输入搬运与本行 V 计算/Pass2/Pass3 重叠（wResident 保证 Pass2/3 不用 inB）。
    pipe_.InitBuffer(bufIn_, static_cast<uint32_t>(6 * K_BLOCK_F * static_cast<int32_t>(sizeof(InT))));
    // 权重驻留：fp16/bf16 且 3*H*4B 权重 + XRow 等放得下 UB 时，gamma/smooth1/smooth2
    // 一次性 cast 成 fp32 常驻 UB，消除每行 Pass2/Pass3 共 6 次 H 元素 GM 搬运 + cast。
    // UB 预算(H=8192, fp16)：3H*4=96K + XRow 32.8K + calc 20.5K + 其他 ~7.3K ≈ 156.6K < 192K。
    // H=16384 时权重 fp32 驻留需 192K 放不下，退回 tile 模式（每 tile 重新加载）。
    wResident_ = (!Std::is_same<InT, float>::value) && (hNum_ <= 8192);
    // 段起始偏移 wSeg*4B 必须 32B 对齐（Cast dst 硬性要求），向上对齐到 8 个 fp32
    const int32_t wSeg = wResident_ ? ((hNum_ + 7) & ~7) : K_BLOCK_F; // 权重段 stride（fp32 元素）
    pipe_.InitBuffer(bufGs_, static_cast<uint32_t>(3 * wSeg * 4));
    pipe_.InitBuffer(bufCalc_, static_cast<uint32_t>(6 * K_BLOCK_F * 4));
    pipe_.InitBuffer(bufOut_, static_cast<uint32_t>(2 * K_BLOCK_F * static_cast<int32_t>(sizeof(InT))));
    pipe_.InitBuffer(bufI8_, static_cast<uint32_t>(2 * K_BLOCK_F));
    // torch 精确归约的级联累加器：acc=[0,64)（4 层 × 16 float），tailBuf=[64,128)（存 H%16 尾部）
    pipe_.InitBuffer(bufAcc_, static_cast<uint32_t>(128 * 4));
    // smooth 半驻留（非驻留路径专用，仅 fp16/bf16）：smooth1/2 以原始 InT 全行存 UB，
    // Pass2/Pass3 逐 tile 从 UB 段 Cast，消除每 tile 2~3 次 GM 读（H=16384 case）。
    // fp32 时 InT=4B 驻留超 UB 预算，保持 GM 逐 tile 读。
    smHalfRes_ = (!Std::is_same<InT, float>::value) && (hNum_ > 8192);
    if (smHalfRes_) {
        smSeg_ = (hNum_ + 15) & ~15; // half/bf16 段起始 32B 对齐（16 元素）
        pipe_.InitBuffer(bufSmH_, static_cast<uint32_t>(2 * smSeg_ * static_cast<int32_t>(sizeof(InT))));
    }
    // bufRed_ 扩到 ROW_FACTOR 行组规模：scale 攒批路径每组 ROW_FACTOR 行 × 2 路 × 4B，
    // +32B 对齐余量。小 B 路径仍只用前 8B（1 行 × 2 路）。
    pipe_.InitBuffer(bufRed_, static_cast<uint32_t>(ROW_FACTOR * 2 * 4 + 32));
    // 输出攒批（大 B 小 H 场景）：x/y/y1/y2 攒 UB 段，组尾 2D DataCopyPad 批量写 GM，
    // 消除逐行 4 次 MTE3 发射 + 12 个事件对。条件：非 fp32 且 h%32==0
    // （Cast dst 32B 对齐 + 2D 搬运 srcStride=0 的块对齐双要求），行数按 UB 剩余自适应，
    // 不足 16 行则关闭（攒批收益摊薄）。布局：batchT=[0,bh*h)x段+[bh*h,2bh*h)y段，
    // batchI8=[0,bh*h)y1段+[bh*h,2bh*h)y2段（行 stride=h，紧凑，srcStride=0）。
    batchRows_ = 0;
    if ((!Std::is_same<InT, float>::value) && (hNum_ % 32) == 0) {
        const int32_t fixedBytes = static_cast<int32_t>(rowBytes + 32u)
            + 6 * K_BLOCK_F * static_cast<int32_t>(sizeof(InT)) + 6 * K_BLOCK_F * 4
            + 2 * K_BLOCK_F * static_cast<int32_t>(sizeof(InT)) + 2 * K_BLOCK_F
            + 3 * wSeg * 4 + (smHalfRes_ ? 2 * smSeg_ * static_cast<int32_t>(sizeof(InT)) : 0)
            + ROW_FACTOR * 2 * 4 + 32;
        const int32_t avail = 176 * 1024 - fixedBytes;
        const int32_t perRow = hNum_ * (2 * static_cast<int32_t>(sizeof(InT)) + 2) + 8;
        if (avail > 0 && perRow > 0) {
            batchRows_ = avail / perRow;
            if (batchRows_ > ROW_FACTOR) {
                batchRows_ = ROW_FACTOR;
            }
            if (batchRows_ < 16) {
                batchRows_ = 0;
            }
        }
    }
    if (batchRows_ > 0) {
        pipe_.InitBuffer(bufBatchT_, static_cast<uint32_t>(2 * batchRows_ * hNum_ * static_cast<int32_t>(sizeof(InT))));
        pipe_.InitBuffer(bufBatchI8_, static_cast<uint32_t>(2 * batchRows_ * hNum_));
    }

    if (wResident_) {
        GlobalTensor<InT> gGamma;
        GlobalTensor<InT> gSm1;
        GlobalTensor<InT> gSm2;
        gGamma.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmGamma_), static_cast<uint64_t>(hNum_));
        gSm1.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmSm1_), static_cast<uint64_t>(hNum_));
        gSm2.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmSm2_), static_cast<uint64_t>(hNum_));
        LocalTensor<InT> inB = bufIn_.Get<InT>();
        LocalTensor<float> gs = bufGs_.Get<float>();
        for (int32_t off = 0; off < hNum_; off += K_BLOCK_F) {
            const int32_t cnt = (hNum_ - off) < K_BLOCK_F ? (hNum_ - off) : K_BLOCK_F;
            Gm2Ub<InT>(inB, gGamma, off, cnt);
            Cast<float>(gs[off], inB, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
            Gm2Ub<InT>(inB, gSm1, off, cnt);
            Cast<float>(gs[wSeg + off], inB, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
            Gm2Ub<InT>(inB, gSm2, off, cnt);
            Cast<float>(gs[2 * wSeg + off], inB, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
        }
        // gs 段此后只被 V pipe 读写（ProcessRow 的 Mul），Cast 写完由上一条 PIPE_V 保序
        PipeBarrier<PIPE_V>();
    }

    if (smHalfRes_) {
        GlobalTensor<InT> gSm1;
        GlobalTensor<InT> gSm2;
        gSm1.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmSm1_), static_cast<uint64_t>(hNum_));
        gSm2.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmSm2_), static_cast<uint64_t>(hNum_));
        LocalTensor<InT> smH = bufSmH_.Get<InT>();
        for (int32_t off = 0; off < hNum_; off += K_BLOCK_F) {
            const int32_t cnt = (hNum_ - off) < K_BLOCK_F ? (hNum_ - off) : K_BLOCK_F;
            Gm2Ub<InT>(smH[off], gSm1, off, cnt);
            Gm2Ub<InT>(smH[smSeg_ + off], gSm2, off, cnt);
        }
    }

    const int64_t blockNum = GetBlockNum();
    // 行间预取模式（fp16/bf16 单 tile 且 wResident）：本行 Cast 读完 inB 后立即发射
    // 下一行全部输入路，MTE2 搬运与本行 Add/归约/Pass2/Pass3 的 V 计算重叠。
    // h<=K_BLOCK_F 时 wResident 必真 → Pass2/3 不读 inB，无预取竞态。
    const bool pfMode = (!Std::is_same<InT, float>::value) && (hNum_ <= K_BLOCK_F);
    LocalTensor<InT> inB = bufIn_.Get<InT>();
    const int32_t h = hNum_;

    const int64_t totalGroups = (batchRows_ > 0) ? ((bNum_ + batchRows_ - 1) / batchRows_) : 0;
    const bool useRowBatch = (batchRows_ > 0) && (totalGroups >= blockNum);
    if (useRowBatch) {
        LocalTensor<float> scaleOut = bufRed_.Get<float>();
        LocalTensor<InT> batchT = bufBatchT_.Get<InT>();
        LocalTensor<int8_t> batchI8 = bufBatchI8_.Get<int8_t>();
        const int32_t bh = batchRows_;
        GlobalTensor<float> gScale1;
        GlobalTensor<float> gScale2;
        gScale1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmScale1_), static_cast<uint64_t>(bNum_));
        gScale2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmScale2_), static_cast<uint64_t>(bNum_));
        for (int64_t g = GetBlockIdx(); g < totalGroups; g += blockNum) {
            const int64_t r0 = g * bh;
            const int64_t r1 = (r0 + bh < bNum_) ? (r0 + bh) : bNum_;
            const int32_t rows = static_cast<int32_t>(r1 - r0);
            if (pfMode) {
                // 组首行输入发射。V_MTE2 必需：首组前 wResident 初始化的最后一次
                // Cast(gs[2*wSeg+off], inB) 是 V 读 inB（PIPE_V 挡不住 MTE2，真机竞态，
                // 第八版 case2-10 WA 根因）；后续组前 V_MTE3 已等全部 V 完成，此处零成本。
                SyncPipe<HardEvent::V_MTE2>();
                LoadRowAsync<InT>(inB, gmX1a_, r0, h);
                LoadRowAsync<InT>(inB[K_BLOCK_F], gmX2_, r0, h);
                if (gmX1b_ != nullptr) {
                    LoadRowAsync<InT>(inB[2 * K_BLOCK_F], gmX1b_, r0, h);
                }
                if (gmX1c_ != nullptr) {
                    LoadRowAsync<InT>(inB[3 * K_BLOCK_F], gmX1c_, r0, h);
                }
                if (gmX1d_ != nullptr) {
                    LoadRowAsync<InT>(inB[4 * K_BLOCK_F], gmX1d_, r0, h);
                }
                if (gmX1e_ != nullptr) {
                    LoadRowAsync<InT>(inB[5 * K_BLOCK_F], gmX1e_, r0, h);
                }
            }
            for (int64_t row = r0; row < r1; ++row) {
                const int64_t nr = (row + 1 < r1) ? (row + 1) : -1;
                ProcessRow<InT>(row, static_cast<int32_t>(row - r0), scaleOut, batchT, batchI8, nr);
            }
            // 组尾：V_MTE3 等 Cast/Round2Int8 写完攒批段；PIPE_ALL 确保 S pipe SetValue 可见
            SyncPipe<HardEvent::V_MTE3>();
            PipeBarrier<PIPE_ALL>();
            {
                GlobalTensor<InT> gXb;
                GlobalTensor<InT> gYb;
                GlobalTensor<int8_t> gY1b;
                GlobalTensor<int8_t> gY2b;
                gXb.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX_) + r0 * h, static_cast<uint64_t>(rows) * h);
                gYb.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmY_) + r0 * h, static_cast<uint64_t>(rows) * h);
                gY1b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(gmY1_) + r0 * h, static_cast<uint64_t>(rows) * h);
                gY2b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(gmY2_) + r0 * h, static_cast<uint64_t>(rows) * h);
                DataCopyExtParams cp;
                cp.blockCount = static_cast<uint32_t>(rows);
                cp.blockLen = static_cast<uint32_t>(h) * static_cast<uint32_t>(sizeof(InT));
                cp.srcStride = 0;
                cp.dstStride = 0;
                DataCopyPad(gXb, batchT, cp);
                DataCopyPad(gYb, batchT[bh * h], cp);
                cp.blockLen = static_cast<uint32_t>(h);
                DataCopyPad(gY1b, batchI8, cp);
                DataCopyPad(gY2b, batchI8[bh * h], cp);
            }
            Ub2GmAsync<float>(gScale1, r0, scaleOut, rows);
            Ub2GmAsync<float>(gScale2, r0, scaleOut[ROW_FACTOR], rows);
            // MTE3 读完攒批段后下一组才能覆写
            SyncPipe<HardEvent::MTE3_V>();
        }
    } else {
        const int64_t rowStart = GetBlockIdx();
        if (pfMode && rowStart < bNum_) {
            // V_MTE2 必需：wResident 初始化最后一次 Cast 读 inB 与本次 MTE2 写 inB 的竞态
            SyncPipe<HardEvent::V_MTE2>();
            LoadRowAsync<InT>(inB, gmX1a_, rowStart, h);
            LoadRowAsync<InT>(inB[K_BLOCK_F], gmX2_, rowStart, h);
            if (gmX1b_ != nullptr) {
                LoadRowAsync<InT>(inB[2 * K_BLOCK_F], gmX1b_, rowStart, h);
            }
            if (gmX1c_ != nullptr) {
                LoadRowAsync<InT>(inB[3 * K_BLOCK_F], gmX1c_, rowStart, h);
            }
            if (gmX1d_ != nullptr) {
                LoadRowAsync<InT>(inB[4 * K_BLOCK_F], gmX1d_, rowStart, h);
            }
            if (gmX1e_ != nullptr) {
                LoadRowAsync<InT>(inB[5 * K_BLOCK_F], gmX1e_, rowStart, h);
            }
        }
        for (int64_t row = rowStart; row < bNum_; row += blockNum) {
            const int64_t nr = (pfMode && row + blockNum < bNum_) ? (row + blockNum) : -1;
            ProcessRow<InT>(row, -1, LocalTensor<float>(), LocalTensor<InT>(), LocalTensor<int8_t>(), nr);
        }
    }
    // kernel 尾兜底：末行 Pass3 末 tile 的 MTE3_V 理论上已等全部 MTE3 排空，
    // 但 async 输出链（x/y 块、scale 双槽）跨行在飞的边界情形（攒批/交错分支切换、
    // 单行 case）无统一尾屏障。PIPE_ALL 确保驱动取走 GM 前一切搬运完成（零成本保险）。
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Gm2UbAsync(LocalTensor<InT> dst, const GlobalTensor<InT>& src, int64_t off, int32_t cnt)
{
    // 纯发射（无同步）：调用方须已保证 dst 上一次 V 读完成（SyncPipe<V_MTE2>）
    // 且发射后须 SyncPipe<MTE2_V> 才能读 dst
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint32_t>(cnt) * static_cast<uint32_t>(sizeof(InT));
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPad(dst, src[off], cp, {});
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::LoadRowAsync(LocalTensor<InT> dst, GM_ADDR base, int64_t row, int32_t cnt)
{
    // 行输入纯发射（无同步）：调用方须已保证 dst 块无在飞 V 读（SyncPipe<V_MTE2> 已过）
    GlobalTensor<InT> g;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(base) + row * hNum_, static_cast<uint64_t>(cnt));
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint32_t>(cnt) * static_cast<uint32_t>(sizeof(InT));
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPad(dst, g[0], cp, {});
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Gm2Ub(LocalTensor<InT> dst, const GlobalTensor<InT>& src, int64_t off, int32_t cnt)
{
    // 与生产参考内核一致：全程 DataCopyPad 字节形式，GM 起始地址无需 32B 对齐。
    // 开头 V_MTE2：等此前 V pipe 指令（上一次 Cast 读 inB / fp32 路径上一行 Muls 写 dst 段）
    // 完成后 MTE2 才能覆写该 UB 段（PIPE_V 挡不住 MTE2，真机实测竞态）。
    SyncPipe<HardEvent::V_MTE2>();
    Gm2UbAsync<InT>(dst, src, off, cnt);
    // 尾部 MTE2_V：等本次拷贝完成后 V 才能读 dst（替代 PIPE_ALL，只等 MTE2→V 方向）
    SyncPipe<HardEvent::MTE2_V>();
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Ub2GmAsync(GlobalTensor<InT>& dst, int64_t off, const LocalTensor<InT>& src, int32_t cnt)
{
    // 纯发射（无同步）：调用方须已保证 src 的 V 写完成（SyncPipe<V_MTE3>），
    // 且该块下次被 V 覆写前须 SyncPipe<MTE3_V>
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint32_t>(cnt) * static_cast<uint32_t>(sizeof(InT));
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPad(dst[off], src, cp);
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::Ub2Gm(GlobalTensor<InT>& dst, int64_t off, const LocalTensor<InT>& src, int32_t cnt)
{
    // 与生产参考内核一致：DataCopyPad 字节形式单次搬运，GM 起始地址无需 32B 对齐，
    // 按 blockLen 精确写入（底层 copy_ubuf_to_gm_align_b32 以 left/right padding
    // 做字节粒度传输，无 32B burst 取整越界）
    SyncPipe<HardEvent::MTE3_V>(); // 该块上一次在飞 MTE3 读完才能被 V 覆写
    Ub2GmAsync<InT>(dst, off, src, cnt);
    // 尾部 MTE3_V：等 MTE3 读完 src 后 V 才能覆写该 UB 段（替代 PIPE_ALL，只等 MTE3→V 方向）
    SyncPipe<HardEvent::MTE3_V>();
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::LoadCastF(LocalTensor<float> dst, const LocalTensor<InT>& inB, const GlobalTensor<InT>& src, int64_t off, int32_t cnt)
{
    if constexpr (Std::is_same<InT, float>::value) {
        Gm2Ub<float>(dst, src, off, cnt);
    } else {
        Gm2Ub<InT>(inB, src, off, cnt);
        Cast<float, InT>(dst, inB, RoundMode::CAST_NONE, cnt);
        // inB 复用保护（Cast 读 inB → 下一次 Gm2Ub 覆写）由 Gm2Ub 开头的 V_MTE2 事件对承担；
        // dst 后续 V 指令读序由 PIPE_V 保证
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline float MultiAddRmsNormDynamicQuant<T>::ReduceSumScalar(const LocalTensor<float>& src, int32_t count)
{
    // 全显式折半二叉树（对齐生产 reduce_common.h::ReduceSumHalfInterval 的
    // findPowerTwo 树形，并显式折到底不再经 WholeReduceSum 收尾）：真机
    // WholeReduceSum 内部归约顺序是黑盒（sim 为树形，真机实现未知），fp32 加法
    // 非结合 → 归约结构差 → rstd ~1ulp 噪声 → int8 翻转。
    // 向量 Add 的 src1 起始须 32B 对齐 → 折半偏移 ≥8 元素，向量链只能折到 8；
    // 最后三级用标量跨半配对树（与向量折半 Add 链逐级同构：先 (i,i+4) 再
    // (i,i+2) 再 (0,1)，树形结构与 torch 向量化归约 golden_probe 实证 0 差异）。
    // Max 幂等无顺序问题（ReduceMaxScalar 保持 WholeReduceMax 收尾）。
    if (count <= 0) {
        return 0.0f;
    }
    if (count < 8) {
        // <8 元素：无对齐安全的向量折叠，标量域同构折半树
        SyncPipe<HardEvent::V_S>();
        float v[7];
        for (int32_t i = 0; i < count; ++i) {
            v[i] = src.GetValue(i);
        }
        int32_t len = count;
        int32_t p = 1;
        while ((p << 1) <= len) {
            p <<= 1;
        }
        const int32_t tail = len - p;
        if (tail > 0) {
            for (int32_t i = 0; i < tail; ++i) {
                v[i] += v[p + i];
            }
            len = p;
        }
        while (len > 1) {
            len >>= 1;
            for (int32_t i = 0; i < len; ++i) {
                v[i] += v[len + i];
            }
        }
        return v[0];
    }
    int32_t n = count;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    int32_t bodyCount = (n + 1) >> 1; // ≤count 的最大 2 幂（findPowerTwo，count≥8 ⇒ ≥8）
    const int32_t tailCount = count - bodyCount;
    if (tailCount > 0) {
        Add(src, src, src[bodyCount], tailCount);
        PipeBarrier<PIPE_V>();
    }
    while (bodyCount > 8) {
        bodyCount >>= 1;
        Add(src, src, src[bodyCount], bodyCount);
        PipeBarrier<PIPE_V>();
    }
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    {
        const float a0 = src.GetValue(0);
        const float a1 = src.GetValue(1);
        const float a2 = src.GetValue(2);
        const float a3 = src.GetValue(3);
        const float a4 = src.GetValue(4);
        const float a5 = src.GetValue(5);
        const float a6 = src.GetValue(6);
        const float a7 = src.GetValue(7);
        const float t0 = a0 + a4;
        const float t1 = a1 + a5;
        const float t2 = a2 + a6;
        const float t3 = a3 + a7;
        return (t0 + t2) + (t1 + t3);
    }
}

template <typename T>
__aicore__ inline float MultiAddRmsNormDynamicQuant<T>::ReduceMaxScalar(const LocalTensor<float>& src, int32_t count)
{
    // 对齐生产 helper.h::ReduceMaxInplace（single_row_kernel::ScaleTensor 真机路径）：
    // repeat 折叠 + 尾部并入 + WholeReduceMax 收尾 + GetValue 仅 1 次。
    // Max 幂等，count<64 时无 Sum 的自翻倍问题。
    constexpr int32_t REP = 64;
    if (count > REP) {
        const int32_t k = count / REP;
        const int32_t rems = count - k * REP;
        if (k > 1) {
            Max(src, src[REP], src, REP, k - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (rems > 0) {
            Max(src, src[k * REP], src, rems);
            PipeBarrier<PIPE_V>();
        }
        WholeReduceMax(src, src, REP, 1, 8, 1, 8);
    } else {
        WholeReduceMax(src, src, count, 1, 8, 1, 8);
    }
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    return src.GetValue(0);
}

template <typename T>
template <typename InT>
__aicore__ inline void MultiAddRmsNormDynamicQuant<T>::ProcessRow(int64_t row, int32_t scaleIdx, LocalTensor<float> scaleOut, LocalTensor<InT> batchT, LocalTensor<int8_t> batchI8, int64_t nextRow)
{
    const int32_t h = hNum_;

    LocalTensor<float> xRow = bufXRow_.Get<float>();
    LocalTensor<InT> inB = bufIn_.Get<InT>();
    LocalTensor<float> gs = bufGs_.Get<float>();
    LocalTensor<float> calc = bufCalc_.Get<float>();
    LocalTensor<InT> outB = bufOut_.Get<InT>();
    LocalTensor<int8_t> i8B = bufI8_.Get<int8_t>();
    LocalTensor<float> red = bufRed_.Get<float>();
    LocalTensor<InT> smH = smHalfRes_ ? bufSmH_.Get<InT>() : LocalTensor<InT>();

    const int32_t wSeg = wResident_ ? ((h + 7) & ~7) : K_BLOCK_F; // 权重段 stride（fp32 元素，32B 对齐）

    GlobalTensor<InT> gX1a;
    GlobalTensor<InT> gX1b;
    GlobalTensor<InT> gX1c;
    GlobalTensor<InT> gX1d;
    GlobalTensor<InT> gX1e;
    GlobalTensor<InT> gX2;
    GlobalTensor<InT> gGamma;
    GlobalTensor<InT> gSm1;
    GlobalTensor<InT> gSm2;
    GlobalTensor<InT> gX;
    GlobalTensor<InT> gY;
    GlobalTensor<int8_t> gY1;
    GlobalTensor<int8_t> gY2;
    GlobalTensor<float> gScale1;
    GlobalTensor<float> gScale2;
    gX1a.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX1a_) + row * h, static_cast<uint64_t>(h));
    if (gmX1b_ != nullptr) {
        gX1b.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX1b_) + row * h, static_cast<uint64_t>(h));
    }
    if (gmX1c_ != nullptr) {
        gX1c.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX1c_) + row * h, static_cast<uint64_t>(h));
    }
    if (gmX1d_ != nullptr) {
        gX1d.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX1d_) + row * h, static_cast<uint64_t>(h));
    }
    if (gmX1e_ != nullptr) {
        gX1e.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX1e_) + row * h, static_cast<uint64_t>(h));
    }
    gX2.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX2_) + row * h, static_cast<uint64_t>(h));
    gGamma.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmGamma_), static_cast<uint64_t>(h));
    gSm1.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmSm1_), static_cast<uint64_t>(h));
    gSm2.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmSm2_), static_cast<uint64_t>(h));
    gX.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmX_) + row * h, static_cast<uint64_t>(h));
    gY.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(gmY_) + row * h, static_cast<uint64_t>(h));
    gY1.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(gmY1_) + row * h, static_cast<uint64_t>(h));
    gY2.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(gmY2_) + row * h, static_cast<uint64_t>(h));
    gScale1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmScale1_), static_cast<uint64_t>(bNum_));
    gScale2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmScale2_), static_cast<uint64_t>(bNum_));

    // torch 精确级联归约状态：accSeg=[0,64)（4 层 × 16 float），tailBuf=[64,128)（H%16 尾部）
    LocalTensor<float> accSeg = bufAcc_.Get<float>();
    LocalTensor<float> tailBuf = accSeg[64];
    Duplicate<float>(accSeg, 0.0f, 64);
    PipeBarrier<PIPE_V>();
    int32_t cascI = 0;
    int32_t tailN = 0;
    float sumSq = 0.0f;
    // pfMode：单 tile + fp16/bf16 + wResident（h<=K_BLOCK_F 时必真）。本行输入已由
    // 组首/上行尾发射（MTE2 与上行 Pass2/3 重叠），此处仅等就绪 → 全路 Cast → 预取下一行。
    const bool pfMode = (!Std::is_same<InT, float>::value) && (h <= K_BLOCK_F);
    if (pfMode) {
        SyncPipe<HardEvent::MTE2_V>();
        Cast<float, InT>(calc[0], inB, RoundMode::CAST_NONE, h);
        Cast<float, InT>(calc[K_BLOCK_F], inB[K_BLOCK_F], RoundMode::CAST_NONE, h);
        if (gmX1b_ != nullptr) {
            Cast<float, InT>(calc[2 * K_BLOCK_F], inB[2 * K_BLOCK_F], RoundMode::CAST_NONE, h);
        }
        if (gmX1c_ != nullptr) {
            Cast<float, InT>(calc[3 * K_BLOCK_F], inB[3 * K_BLOCK_F], RoundMode::CAST_NONE, h);
        }
        if (gmX1d_ != nullptr) {
            Cast<float, InT>(calc[4 * K_BLOCK_F], inB[4 * K_BLOCK_F], RoundMode::CAST_NONE, h);
        }
        if (gmX1e_ != nullptr) {
            Cast<float, InT>(calc[5 * K_BLOCK_F], inB[5 * K_BLOCK_F], RoundMode::CAST_NONE, h);
        }
        if (nextRow >= 0) {
            // 全部 Cast 读完 inB 后立即发射下一行输入（MTE2 与后续 V 计算重叠）。
            // V FIFO 保证 Cast→此处无 RAW 屏障必要（同 pipe 指令序）。
            SyncPipe<HardEvent::V_MTE2>();
            LoadRowAsync<InT>(inB, gmX1a_, nextRow, h);
            LoadRowAsync<InT>(inB[K_BLOCK_F], gmX2_, nextRow, h);
            if (gmX1b_ != nullptr) {
                LoadRowAsync<InT>(inB[2 * K_BLOCK_F], gmX1b_, nextRow, h);
            }
            if (gmX1c_ != nullptr) {
                LoadRowAsync<InT>(inB[3 * K_BLOCK_F], gmX1c_, nextRow, h);
            }
            if (gmX1d_ != nullptr) {
                LoadRowAsync<InT>(inB[4 * K_BLOCK_F], gmX1d_, nextRow, h);
            }
            if (gmX1e_ != nullptr) {
                LoadRowAsync<InT>(inB[5 * K_BLOCK_F], gmX1e_, nextRow, h);
            }
        }
        // Add 链保持金标顺序：x1a+x2 → +x1b → +x1c → +x1d → +x1e（Cast 提前不改数值）
        Add(xRow[0], calc[0], calc[K_BLOCK_F], h);
        if (gmX1b_ != nullptr) {
            Add(xRow[0], xRow[0], calc[2 * K_BLOCK_F], h);
        }
        if (gmX1c_ != nullptr) {
            Add(xRow[0], xRow[0], calc[3 * K_BLOCK_F], h);
        }
        if (gmX1d_ != nullptr) {
            Add(xRow[0], xRow[0], calc[4 * K_BLOCK_F], h);
        }
        if (gmX1e_ != nullptr) {
            Add(xRow[0], xRow[0], calc[5 * K_BLOCK_F], h);
        }
        Mul(calc[2 * K_BLOCK_F], xRow[0], xRow[0], h);
        TorchCascadeAdd(accSeg, calc[2 * K_BLOCK_F], h / 16, cascI);
        {
            const int32_t tb = (h / 16) * 16;
            tailN = h - tb;
            SyncPipe<HardEvent::V_S>();
            for (int32_t j = 0; j < tailN; ++j) {
                tailBuf.SetValue(j, calc[2 * K_BLOCK_F].GetValue(tb + j));
            }
        }
        if (scaleIdx >= 0) {
            // 输出攒批：Cast 到组缓冲 x 段（rg*h 偏移，h%32==0 保证 32B 对齐），组尾 2D 批写
            Cast<InT, float>(batchT[static_cast<int64_t>(scaleIdx) * h], xRow[0], RoundMode::CAST_RINT, h);
        } else {
            // outB 块0=x 专用（块1=y，Pass2 用）；MTE3_V 等块0 上次在飞搬运读完。
            // Cast→V_MTE3→Ub2GmAsync：比同步 Ub2Gm（头尾双 MTE3_V）省 1 个事件对。
            SyncPipe<HardEvent::MTE3_V>();
            Cast<InT, float>(outB, xRow[0], RoundMode::CAST_RINT, h);
            SyncPipe<HardEvent::V_MTE3>();
            Ub2GmAsync<InT>(gX, 0, outB, h);
        }
    } else {
    for (int32_t off = 0; off < h; off += K_BLOCK_F) {
        const int32_t cnt = (h - off) < K_BLOCK_F ? (h - off) : K_BLOCK_F;
        if constexpr (Std::is_same<InT, float>::value) {
            // fp32 保持 LoadCastF 旧路径（probe 实测 fp32 直读 inB 的 Add 有数值问题，另查）
            LoadCastF<InT>(calc[0], inB, gX1a, off, cnt);
            LoadCastF<InT>(calc[K_BLOCK_F], inB, gX2, off, cnt);
            Add(xRow[off], calc[0], calc[K_BLOCK_F], cnt);
            PipeBarrier<PIPE_V>();
            if (gmX1b_ != nullptr) {
                LoadCastF<InT>(calc[K_BLOCK_F], inB, gX1b, off, cnt);
                Add(xRow[off], xRow[off], calc[K_BLOCK_F], cnt);
                PipeBarrier<PIPE_V>();
            }
            if (gmX1c_ != nullptr) {
                LoadCastF<InT>(calc[K_BLOCK_F], inB, gX1c, off, cnt);
                Add(xRow[off], xRow[off], calc[K_BLOCK_F], cnt);
                PipeBarrier<PIPE_V>();
            }
            if (gmX1d_ != nullptr) {
                LoadCastF<InT>(calc[K_BLOCK_F], inB, gX1d, off, cnt);
                Add(xRow[off], xRow[off], calc[K_BLOCK_F], cnt);
                PipeBarrier<PIPE_V>();
            }
            if (gmX1e_ != nullptr) {
                LoadCastF<InT>(calc[K_BLOCK_F], inB, gX1e, off, cnt);
                Add(xRow[off], xRow[off], calc[K_BLOCK_F], cnt);
                PipeBarrier<PIPE_V>();
            }
            Mul(calc[2 * K_BLOCK_F], xRow[off], xRow[off], cnt);
            PipeBarrier<PIPE_V>();
            TorchCascadeAdd(accSeg, calc[2 * K_BLOCK_F], cnt / 16, cascI);
            if (off + cnt == h) {
                const int32_t tbOff = (h / 16) * 16 - off;
                tailN = h - (h / 16) * 16;
                SyncPipe<HardEvent::V_S>();
                for (int32_t j = 0; j < tailN; ++j) {
                    tailBuf.SetValue(j, calc[2 * K_BLOCK_F].GetValue(tbOff + j));
                }
            }
            SyncPipe<HardEvent::V_MTE3>();
            Ub2Gm<InT>(gX, off, xRow[off], cnt);
        } else {
            // 6 路并行发射（与 pfMode 同模式）：一次 V_MTE2 + 全路 MTE2 + 一次 MTE2_V
            // + 全路 Cast + Add 链。事件对 2+2n → 2/tile；Add 链保持金标顺序
            // （x1a+x2→+x1b→…→+x1e，Cast 提前不改数值，V FIFO 保证 Cast→Add 读序）。
            SyncPipe<HardEvent::V_MTE2>();
            Gm2UbAsync<InT>(inB, gX1a, off, cnt);
            Gm2UbAsync<InT>(inB[K_BLOCK_F], gX2, off, cnt);
            if (gmX1b_ != nullptr) {
                Gm2UbAsync<InT>(inB[2 * K_BLOCK_F], gX1b, off, cnt);
            }
            if (gmX1c_ != nullptr) {
                Gm2UbAsync<InT>(inB[3 * K_BLOCK_F], gX1c, off, cnt);
            }
            if (gmX1d_ != nullptr) {
                Gm2UbAsync<InT>(inB[4 * K_BLOCK_F], gX1d, off, cnt);
            }
            if (gmX1e_ != nullptr) {
                Gm2UbAsync<InT>(inB[5 * K_BLOCK_F], gX1e, off, cnt);
            }
            SyncPipe<HardEvent::MTE2_V>();
            Cast<float, InT>(calc[0], inB, RoundMode::CAST_NONE, cnt);
            Cast<float, InT>(calc[K_BLOCK_F], inB[K_BLOCK_F], RoundMode::CAST_NONE, cnt);
            if (gmX1b_ != nullptr) {
                Cast<float, InT>(calc[2 * K_BLOCK_F], inB[2 * K_BLOCK_F], RoundMode::CAST_NONE, cnt);
            }
            if (gmX1c_ != nullptr) {
                Cast<float, InT>(calc[3 * K_BLOCK_F], inB[3 * K_BLOCK_F], RoundMode::CAST_NONE, cnt);
            }
            if (gmX1d_ != nullptr) {
                Cast<float, InT>(calc[4 * K_BLOCK_F], inB[4 * K_BLOCK_F], RoundMode::CAST_NONE, cnt);
            }
            if (gmX1e_ != nullptr) {
                Cast<float, InT>(calc[5 * K_BLOCK_F], inB[5 * K_BLOCK_F], RoundMode::CAST_NONE, cnt);
            }
            Add(xRow[off], calc[0], calc[K_BLOCK_F], cnt);
            if (gmX1b_ != nullptr) {
                Add(xRow[off], xRow[off], calc[2 * K_BLOCK_F], cnt);
            }
            if (gmX1c_ != nullptr) {
                Add(xRow[off], xRow[off], calc[3 * K_BLOCK_F], cnt);
            }
            if (gmX1d_ != nullptr) {
                Add(xRow[off], xRow[off], calc[4 * K_BLOCK_F], cnt);
            }
            if (gmX1e_ != nullptr) {
                Add(xRow[off], xRow[off], calc[5 * K_BLOCK_F], cnt);
            }
            Mul(calc[2 * K_BLOCK_F], xRow[off], xRow[off], cnt);
            PipeBarrier<PIPE_V>();
            TorchCascadeAdd(accSeg, calc[2 * K_BLOCK_F], cnt / 16, cascI);
            if (off + cnt == h) {
                const int32_t tbOff = (h / 16) * 16 - off;
                tailN = h - (h / 16) * 16;
                SyncPipe<HardEvent::V_S>();
                for (int32_t j = 0; j < tailN; ++j) {
                    tailBuf.SetValue(j, calc[2 * K_BLOCK_F].GetValue(tbOff + j));
                }
            }
            // x 输出走 outB 块0 + async（同 pfMode 注释）
            SyncPipe<HardEvent::MTE3_V>();
            Cast<InT, float>(outB, xRow[off], RoundMode::CAST_RINT, cnt);
            SyncPipe<HardEvent::V_MTE3>();
            Ub2GmAsync<InT>(gX, off, outB, cnt);
        }
    }
    }

    sumSq = TorchCascadeFinal(accSeg, tailBuf, h);
    float rstd = 0.0f;
    // rstd = 1.0/sqrt(mean+eps)：910B 无 IEEE FMA（vmadd 为两步舍入），
    // 用 Dekker TwoProd（12-bit split，乘积精确）求精确残差，Newton 精化 sqrt 与 1/x。
    {
        LocalTensor<float> rstdBuf = calc;
        Duplicate<float>(rstdBuf[0], sumSq, 1);
        Duplicate<float>(rstdBuf[8], static_cast<float>(h), 1);
        // mean = sumSq/h 正确舍入除法（Dekker 内联；h 非 2 幂时 vdiv 近似差 1ulp）
        IeeeDiv(rstdBuf[16], rstdBuf[0], rstdBuf[8], 1);      // q0
        Mul(rstdBuf[40], rstdBuf[8], rstdBuf[16], 1);         // p = h*q0
        Muls(rstdBuf[48], rstdBuf[16], 4097.0f, 1);
        Sub(rstdBuf[56], rstdBuf[48], rstdBuf[16], 1);
        Sub(rstdBuf[64], rstdBuf[48], rstdBuf[56], 1);        // qh
        Sub(rstdBuf[72], rstdBuf[16], rstdBuf[64], 1);        // ql
        Muls(rstdBuf[48], rstdBuf[8], 4097.0f, 1);
        Sub(rstdBuf[56], rstdBuf[48], rstdBuf[8], 1);
        Sub(rstdBuf[80], rstdBuf[48], rstdBuf[56], 1);        // bh
        Sub(rstdBuf[88], rstdBuf[8], rstdBuf[80], 1);         // bl
        Mul(rstdBuf[96], rstdBuf[64], rstdBuf[80], 1);
        Sub(rstdBuf[96], rstdBuf[96], rstdBuf[40], 1);        // qh*bh - p
        Mul(rstdBuf[104], rstdBuf[64], rstdBuf[88], 1);
        Add(rstdBuf[96], rstdBuf[96], rstdBuf[104], 1);
        Mul(rstdBuf[104], rstdBuf[72], rstdBuf[80], 1);
        Add(rstdBuf[96], rstdBuf[96], rstdBuf[104], 1);
        Mul(rstdBuf[104], rstdBuf[72], rstdBuf[88], 1);
        Add(rstdBuf[96], rstdBuf[96], rstdBuf[104], 1);       // err
        Sub(rstdBuf[104], rstdBuf[0], rstdBuf[40], 1);        // sumSq - p
        Sub(rstdBuf[104], rstdBuf[104], rstdBuf[96], 1);      // e
        IeeeDiv(rstdBuf[112], rstdBuf[104], rstdBuf[8], 1);   // corr
        Add(rstdBuf[16], rstdBuf[16], rstdBuf[112], 1);       // mean = q0+corr
        Duplicate<float>(rstdBuf[24], eps_, 1);
        Add<float>(rstdBuf[32], rstdBuf[16], rstdBuf[24], 1); // x
        IeeeSqrt(rstdBuf[40], rstdBuf[32], 1);                // s0
        Duplicate<float>(rstdBuf[56], 0.0f, 1);
        Add(rstdBuf[56], rstdBuf[40], rstdBuf[56], 1);        // s0 备份
        // ---- sqrt 精化（Dekker Newton）----
        Mul(rstdBuf[64], rstdBuf[40], rstdBuf[40], 1);
        Muls(rstdBuf[72], rstdBuf[40], 4097.0f, 1);
        Sub(rstdBuf[80], rstdBuf[72], rstdBuf[40], 1);
        Sub(rstdBuf[88], rstdBuf[72], rstdBuf[80], 1);        // ah
        Sub(rstdBuf[96], rstdBuf[40], rstdBuf[88], 1);        // al
        Mul(rstdBuf[104], rstdBuf[88], rstdBuf[88], 1);
        Sub(rstdBuf[104], rstdBuf[104], rstdBuf[64], 1);
        Mul(rstdBuf[112], rstdBuf[88], rstdBuf[96], 1);
        Add(rstdBuf[104], rstdBuf[104], rstdBuf[112], 1);
        Add(rstdBuf[104], rstdBuf[104], rstdBuf[112], 1);
        Mul(rstdBuf[112], rstdBuf[96], rstdBuf[96], 1);
        Add(rstdBuf[120], rstdBuf[104], rstdBuf[112], 1);     // err
        Sub(rstdBuf[136], rstdBuf[32], rstdBuf[64], 1);       // x - p
        Sub(rstdBuf[136], rstdBuf[136], rstdBuf[120], 1);     // t
        Muls(rstdBuf[128], rstdBuf[56], 2.0f, 1);
        IeeeDiv(rstdBuf[144], rstdBuf[136], rstdBuf[128], 1);
        Add(rstdBuf[40], rstdBuf[56], rstdBuf[144], 1);       // s
        // ---- 1/s 精化（Dekker Newton）----
        Duplicate<float>(rstdBuf[48], 1.0f, 1);
        IeeeDiv(rstdBuf[56], rstdBuf[48], rstdBuf[40], 1);    // q0
        Mul(rstdBuf[64], rstdBuf[40], rstdBuf[56], 1);        // p = s*q0
        Muls(rstdBuf[72], rstdBuf[40], 4097.0f, 1);
        Sub(rstdBuf[80], rstdBuf[72], rstdBuf[40], 1);
        Sub(rstdBuf[88], rstdBuf[72], rstdBuf[80], 1);        // ah
        Sub(rstdBuf[96], rstdBuf[40], rstdBuf[88], 1);        // al
        Muls(rstdBuf[104], rstdBuf[56], 4097.0f, 1);
        Sub(rstdBuf[112], rstdBuf[104], rstdBuf[56], 1);
        Sub(rstdBuf[120], rstdBuf[104], rstdBuf[112], 1);     // bh
        Sub(rstdBuf[128], rstdBuf[56], rstdBuf[120], 1);      // bl
        Mul(rstdBuf[136], rstdBuf[88], rstdBuf[120], 1);
        Sub(rstdBuf[136], rstdBuf[136], rstdBuf[64], 1);
        Mul(rstdBuf[144], rstdBuf[88], rstdBuf[128], 1);
        Add(rstdBuf[136], rstdBuf[136], rstdBuf[144], 1);
        Mul(rstdBuf[144], rstdBuf[96], rstdBuf[120], 1);
        Add(rstdBuf[136], rstdBuf[136], rstdBuf[144], 1);
        Mul(rstdBuf[144], rstdBuf[96], rstdBuf[128], 1);
        Add(rstdBuf[136], rstdBuf[136], rstdBuf[144], 1);     // err
        Sub(rstdBuf[152], rstdBuf[48], rstdBuf[64], 1);       // 1 - p
        Sub(rstdBuf[152], rstdBuf[152], rstdBuf[136], 1);     // e
        IeeeDiv(rstdBuf[160], rstdBuf[152], rstdBuf[40], 1);  // corr
        Add(rstdBuf[56], rstdBuf[56], rstdBuf[160], 1);       // r = q0 + corr
        SyncPipe<HardEvent::V_S>();
        rstd = rstdBuf.GetValue(56);
    }
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV);
    WaitFlag<HardEvent::S_V>(eventSV);

    float rowMax1 = 0.0f;
    float rowMax2 = 0.0f;
    for (int32_t off = 0; off < h; off += K_BLOCK_F) {
        const int32_t cnt = (h - off) < K_BLOCK_F ? (h - off) : K_BLOCK_F;
        LocalTensor<float> gW0;
        LocalTensor<float> gW1;
        LocalTensor<float> gW2;
        if (wResident_) {
            gW0 = gs[off];
            gW1 = gs[wSeg + off];
            gW2 = gs[2 * wSeg + off];
        } else if (smHalfRes_) {
            // smooth 半驻留：从 UB 的原始 InT 段 Cast（免 GM 读）；gamma 仍 GM 读
            LoadCastF<InT>(gs[0], inB, gGamma, off, cnt);
            Cast<float, InT>(gs[K_BLOCK_F], smH[off], RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
            Cast<float, InT>(gs[2 * K_BLOCK_F], smH[smSeg_ + off], RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
            gW0 = gs[0];
            gW1 = gs[K_BLOCK_F];
            gW2 = gs[2 * K_BLOCK_F];
        } else {
            LoadCastF<InT>(gs[0], inB, gGamma, off, cnt);
            LoadCastF<InT>(gs[K_BLOCK_F], inB, gSm1, off, cnt);
            LoadCastF<InT>(gs[2 * K_BLOCK_F], inB, gSm2, off, cnt);
            gW0 = gs[0];
            gW1 = gs[K_BLOCK_F];
            gW2 = gs[2 * K_BLOCK_F];
        }
        Muls(xRow[off], xRow[off], rstd, cnt);
        PipeBarrier<PIPE_V>();
        // z 原地化：xRow 从 x 变为 z（x*rstd*gamma），Pass3 直接复用免重算 Muls+Mul
        Mul(xRow[off], xRow[off], gW0, cnt);
        PipeBarrier<PIPE_V>();
        if (scaleIdx >= 0) {
            // 输出攒批：y 段 = batchT 偏移 bh*h + rg*h，组尾 2D 批写 GM
            Cast<InT, float>(batchT[static_cast<int64_t>(batchRows_) * h + static_cast<int64_t>(scaleIdx) * h],
                             xRow[off], RoundMode::CAST_RINT, cnt);
        } else if constexpr (Std::is_same<InT, float>::value) {
            SyncPipe<HardEvent::V_MTE3>();
            Ub2Gm<InT>(gY, off, xRow[off], cnt);
        } else {
            // outB 块1=y 专用（块0=x，Pass1 用）：与 x 输出互不等待；MTE3_V 等块1
            // 上次在飞搬运读完（上一 tile 的 y / 上一行 Pass2）
            SyncPipe<HardEvent::MTE3_V>();
            Cast<InT, float>(outB[K_BLOCK_F], xRow[off], RoundMode::CAST_RINT, cnt);
            SyncPipe<HardEvent::V_MTE3>();
            Ub2GmAsync<InT>(gY, off, outB[K_BLOCK_F], cnt);
        }
        Mul(calc[3 * K_BLOCK_F], xRow[off], gW1, cnt);
        PipeBarrier<PIPE_V>();
        Mul(calc[4 * K_BLOCK_F], xRow[off], gW2, cnt);
        PipeBarrier<PIPE_V>();
        Abs(calc[3 * K_BLOCK_F], calc[3 * K_BLOCK_F], cnt);
        PipeBarrier<PIPE_V>();
        Abs(calc[4 * K_BLOCK_F], calc[4 * K_BLOCK_F], cnt);
        PipeBarrier<PIPE_V>();
        const float max1 = ReduceMaxScalar(calc[3 * K_BLOCK_F], cnt);
        const float max2 = ReduceMaxScalar(calc[4 * K_BLOCK_F], cnt);
        if (max1 > rowMax1) {
            rowMax1 = max1;
        }
        if (max2 > rowMax2) {
            rowMax2 = max2;
        }
    }

    // scale = max/127.0（金标 torch 真除法，IEEE 正确舍入）。旧实现标量
    // rowMax/127.0f 在真机标量单元可能快速近似（十四版实证标量 sqrt()/1.0f/
    // 同病）或被编译成乘倒数 → scale 差 ~1ulp → q/scale 全行漂移 → int8 翻转
    // ——十五版修复：六版以来 case12-15 恒 WA 的最后残留差异源。
    // 向量化 IeeeDiv（段独立 dst：[0]=max1 [8]=127 [16]=scale1 [24]=max2
    // [32]=scale2，calc[0..64) 的 rstd 旧值已标量化可覆写）。
    // 写回仍经 UB + DataCopyPad（MTE3 通道）：GM SetValue 真机不可靠（生产无一处
    // GM SetValue，全走 UB+DataCopy；旧实现 SetValue 直写 GM 曾致 scale 输出全错）。
    float scale1 = 0.0f;
    float scale2 = 0.0f;
    {
        LocalTensor<float> sBuf = calc;
        Duplicate<float>(sBuf[0], rowMax1, 1);
        Duplicate<float>(sBuf[8], 127.0f, 1);
        TorchDiv(sBuf[16], sBuf[0], sBuf[8], 1, bufAcc_.Get<float>());
        Duplicate<float>(sBuf[24], rowMax2, 1);
        TorchDiv(sBuf[32], sBuf[24], sBuf[8], 1, bufAcc_.Get<float>());
        SyncPipe<HardEvent::V_S>();
        scale1 = sBuf.GetValue(16);
        scale2 = sBuf.GetValue(32);
    }
    if (scaleIdx >= 0) {
        // 攒批模式：SetValue 写 UB 段（S pipe），组尾 PIPE_ALL + 批量 MTE3 写 GM。
        // 对齐生产 single_row_kernel::ScaleTensor 的 scaleTensor.SetValue(idx, ...) 模式。
        // bufRed_ 布局：[0, ROW_FACTOR) = scale1，[ROW_FACTOR, 2*ROW_FACTOR) = scale2。
        scaleOut.SetValue(scaleIdx, scale1);
        scaleOut.SetValue(scaleIdx + ROW_FACTOR, scale2);
    } else {
        // A/B 实验回退段（十一版）：scale 回退同步 Ub2Gm（九版已验证模式）。
        // 十版双槽 async 在 case12-15（多 tile 大数据量）疑似概率性竞态致极小错误率
        // （<0.005% 显示 0.00%），x/y 输出 async 保留（pfMode+case11 双重真机验证）。
        Duplicate<float>(red, scale1, 1);
        SyncPipe<HardEvent::V_MTE3>();
        Ub2Gm<float>(gScale1, static_cast<int64_t>(row), red, 1);
        Duplicate<float>(red, scale2, 1);
        SyncPipe<HardEvent::V_MTE3>();
        Ub2Gm<float>(gScale2, static_cast<int64_t>(row), red, 1);
    }

    const float div1 = (scale1 > 0.0f) ? scale1 : 1.0f;
    const float div2 = (scale2 > 0.0f) ? scale2 : 1.0f;
    // div1/div2 的 Dekker split（bh/bl，向量化 count=1 计算避免标量单元非 IEEE）。
    // calc 段布局（bufCalc_ 6144 float）：[0,1K)q [1K,2K)q/s [2K,3328)TorchDivC w
    //   [3328,4416)RNE w [4416,4672)div 广播 [4672,4928)bh [4928,5184)bl（路间覆写）。
    float bh1 = 0.0f;
    float bl1 = 0.0f;
    float bh2 = 0.0f;
    float bl2 = 0.0f;
    {
        LocalTensor<float> sb = calc[2 * K_BLOCK_F]; // 借 TorchDivC w 段头（此时尚未用）
        Duplicate<float>(sb[0], div1, 1);
        Muls(sb[8], sb[0], 4097.0f, 1);
        Sub(sb[16], sb[8], sb[0], 1);
        Sub(sb[24], sb[8], sb[16], 1);               // bh1
        Sub(sb[32], sb[0], sb[24], 1);               // bl1
        Duplicate<float>(sb[64], div2, 1);
        Muls(sb[72], sb[64], 4097.0f, 1);
        Sub(sb[80], sb[72], sb[64], 1);
        Sub(sb[88], sb[72], sb[80], 1);              // bh2
        Sub(sb[96], sb[64], sb[88], 1);              // bl2
        SyncPipe<HardEvent::V_S>();
        bh1 = sb.GetValue(24);
        bl1 = sb.GetValue(32);
        bh2 = sb.GetValue(88);
        bl2 = sb.GetValue(96);
    }
    // 量化除法必须用真除法（Div）：金标 torch.round(q/scale) 是 IEEE fp32 除法；
    // 旧实现 Muls(q, 1/scale) 两次除法舍入的倒数与一次除法差 ~1ulp，
    // q/s 恰在 rint 边界（x.5）时翻转 1 LSB → int8 逐位比对（社区标准 bitwise，
    // 无 LSB 容差）失败 → 错误率 ~1e-5 → 大 case（3~35M 元素）必碰撞 →
    // 占比显示 0.00% 但 WA（case12-15 自六版起的根因；v8 实测 i8Noise 印证）。
    // 广播除数用 calc[5K] 段（Pass3 时 Pass1/Pass2 的旧数据已用完，V FIFO 保序）。
    for (int32_t off = 0; off < h; off += K_BLOCK_F) {
        const int32_t cnt = (h - off) < K_BLOCK_F ? (h - off) : K_BLOCK_F;
        // xRow[off] 已是 z（Pass2 原地化），此处免 Muls(rstd)+Mul(gW0) 重算
        LocalTensor<float> gW1;
        LocalTensor<float> gW2;
        if (wResident_) {
            gW1 = gs[wSeg + off];
            gW2 = gs[2 * wSeg + off];
        } else if (smHalfRes_) {
            Cast<float, InT>(gs[0], smH[off], RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
            Cast<float, InT>(gs[K_BLOCK_F], smH[smSeg_ + off], RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
            gW1 = gs[0];
            gW2 = gs[K_BLOCK_F];
        } else {
            LoadCastF<InT>(gs[0], inB, gSm1, off, cnt);
            LoadCastF<InT>(gs[K_BLOCK_F], inB, gSm2, off, cnt);
            gW1 = gs[0];
            gW2 = gs[K_BLOCK_F];
        }
        // q → calc[0]、q/scale → calc[1K]（十五版 dst 独立化）：0ulp Div 可能多
        // pass 迭代，dst 原地覆写 src 会破坏后续 pass 读值。calc[0]/[1K] 在 Pass3
        // 空闲（Pass1 旧值已用完），calc[5K] 保持广播除数段（div1→div2 轮流覆写，
        // V FIFO 保序）。
        Mul(calc[0], xRow[off], gW1, cnt);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(calc[4416], div1, 256);
        Duplicate<float>(calc[4672], bh1, 256);
        Duplicate<float>(calc[4928], bl1, 256);
        for (int32_t s = 0; s < cnt; s += 256) {
            const int32_t sc = (cnt - s) < 256 ? (cnt - s) : 256;
            TorchDivC(calc[K_BLOCK_F][s], calc[0][s], calc[4416], calc[4672], calc[4928], sc,
                      calc[2 * K_BLOCK_F]);
        }
        PipeBarrier<PIPE_V>();
        if (scaleIdx >= 0) {
            // 输出攒批：y1 段 = batchI8[rg*h]，无逐行 V_MTE3+Ub2Gm（组尾 V_MTE3 统一同步）
            Round2Int8(calc[K_BLOCK_F], batchI8[static_cast<int64_t>(scaleIdx) * h], cnt, calc[3328]);
            Mul(calc[0], xRow[off], gW2, cnt);
            PipeBarrier<PIPE_V>();
            Duplicate<float>(calc[4416], div2, 256);
            Duplicate<float>(calc[4672], bh2, 256);
            Duplicate<float>(calc[4928], bl2, 256);
            for (int32_t s = 0; s < cnt; s += 256) {
            const int32_t sc = (cnt - s) < 256 ? (cnt - s) : 256;
            TorchDivC(calc[K_BLOCK_F][s], calc[0][s], calc[4416], calc[4672], calc[4928], sc,
                      calc[2 * K_BLOCK_F]);
        }
            PipeBarrier<PIPE_V>();
            Round2Int8(calc[K_BLOCK_F],
                       batchI8[static_cast<int64_t>(batchRows_) * h + static_cast<int64_t>(scaleIdx) * h], cnt, calc[3328]);
        } else {
            // A/B 实验回退段（十一版）：i8B 回退单块 + 同步 Ub2Gm（九版已验证模式，
            // 头尾 MTE3_V 内置于 Ub2Gm）。十版双块 async 的 tile 尾集中等待在
            // case12-15（多 tile 大数据量）疑似概率性竞态。
            Round2Int8(calc[K_BLOCK_F], i8B, cnt, calc[3328]);
            SyncPipe<HardEvent::V_MTE3>();
            Ub2Gm<int8_t>(gY1, off, i8B, cnt);
            Mul(calc[0], xRow[off], gW2, cnt);
            PipeBarrier<PIPE_V>();
            Duplicate<float>(calc[4416], div2, 256);
            Duplicate<float>(calc[4672], bh2, 256);
            Duplicate<float>(calc[4928], bl2, 256);
            for (int32_t s = 0; s < cnt; s += 256) {
            const int32_t sc = (cnt - s) < 256 ? (cnt - s) : 256;
            TorchDivC(calc[K_BLOCK_F][s], calc[0][s], calc[4416], calc[4672], calc[4928], sc,
                      calc[2 * K_BLOCK_F]);
        }
            PipeBarrier<PIPE_V>();
            Round2Int8(calc[K_BLOCK_F], i8B, cnt, calc[3328]);
            SyncPipe<HardEvent::V_MTE3>();
            Ub2Gm<int8_t>(gY2, off, i8B, cnt);
        }
    }
}

} // namespace NsMultiAddRmsNormDynamicQuant
#endif // MULTIADDRMSNORMDYNAMICQUANT_H
