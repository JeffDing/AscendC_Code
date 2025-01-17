/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t blockLength,
                              uint32_t tileNum, uint32_t tileLength,
                              uint32_t lasttileLength, uint32_t formerNum,
                              uint32_t formerLength, uint32_t formertileNum,
                              uint32_t formertileLength,
                              uint32_t formerlasttileLength, uint32_t tailNum,
                              uint32_t tailLength, uint32_t tailtileNum,
                              uint32_t tailtileLength,
                              uint32_t taillasttileLength, uint32_t tilingKey)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        if (tilingKey == 1) {
          this->blockLength = blockLength;
          this->tileNum =
              tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
          this->tileLength = tileLength / BUFFER_NUM;
          this->lasttileLength = lasttileLength;
          xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(),
                              this->blockLength);
          zGm.SetGlobalBuffer(
              (__gm__ half*)z + this->blockLength * GetBlockIdx(),
              this->blockLength);
        }

        if (tilingKey == 2) {
          this->formerNum = formerNum;
          this->formerLength = formerLength;
          this->formertileNum = formertileNum;
          this->formertileLength = formertileLength;
          this->formerlasttileLength = formerlasttileLength;

          this->tailNum = tailNum;
          this->tailLength = tailLength;
          this->tailtileNum = tailtileNum;
          this->tailtileLength = tailtileLength;
          this->taillasttileLength = taillasttileLength;

          if (GetBlockIdx() < this->formerNum) {  //分到大块核的处理
            this->tileLength = this->formertileLength / BUFFER_NUM;
            this->lasttileLength = this->formerlasttileLength;
            this->tileNum = this->formertileNum * BUFFER_NUM;
            xGm.SetGlobalBuffer(
                (__gm__ half*)x + this->formerLength * GetBlockIdx(),
                this->formerLength);
            zGm.SetGlobalBuffer(
                (__gm__ half*)z + this->formerLength * GetBlockIdx(),
                this->formerLength);
          } else {  //分到小块核的处理，需要处理的数据量比大核少alignNum个
            this->tileLength = this->tailtileLength / BUFFER_NUM;
            this->lasttileLength = this->taillasttileLength;
            this->tileNum = this->tailtileNum * BUFFER_NUM;
            xGm.SetGlobalBuffer(
                (__gm__ half*)x + this->formerLength * this->formerNum +
                    this->tailLength * (GetBlockIdx() - this->formerNum),
                this->tailLength);
            zGm.SetGlobalBuffer(
                (__gm__ half*)z + this->formerLength * this->formerNum +
                    this->tailLength * (GetBlockIdx() - this->formerNum),
                this->tailLength);
          }
        } 
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> inLocal = inQueueX.AllocTensor<half>();
        
        if (BUFFER_NUM == 1) {
              if (progress == this->tileNum - 1) {
                if (progress == 0) {
                  //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
                  DataCopy(inLocal[0], xGm[0], this->tileLength);
                } else {
                  //将最后一个分块的起始地址向前移动tileLength-lasttileLength
                  DataCopy(
                      inLocal[0],
                      xGm[(progress - 1) * this->tileLength + this->lasttileLength],
                      this->tileLength);
                }
              } else {
                DataCopy(inLocal[0], xGm[progress * this->tileLength],
                         this->tileLength);
              }
            }
        if (BUFFER_NUM == 2) {
          //开启double
          //buffer时，由于将输入数据分成了相等的2部分，分块大小为不开启double
          //buffer的一半， 所以需要对最后两个分块数据的起始地址做处理
          if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
              (progress == (this->tileNum * BUFFER_NUM - 1))) {
            //分块大小变为tileLength的一半
            //倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动
            DataCopy(
                inLocal[0],
                xGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
                (this->tileLength));
          }

          else {
            DataCopy(inLocal[0], xGm[progress * (this->tileLength)],
                     (this->tileLength));
          }
        }
        
        inQueueX.EnQue(inLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {  
        LocalTensor<half> inLocal = inQueueX.DeQue<half>();
        LocalTensor<half> xLocal = inLocal;
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Exp(xLocal, xLocal, this->tileLength);
        Reciprocal(zLocal, xLocal, this->tileLength);
        Sub(zLocal, xLocal, zLocal, this->tileLength);
        half scalar = 0.5;
	Muls(zLocal, zLocal, scalar, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(inLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        if (BUFFER_NUM == 1) {
          if (progress == this->tileNum - 1) {
            if (progress == 0) {
              //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
              DataCopy(zGm[0], zLocal, this->tileLength);
            } else {
              //将最后一个分块的起始地址向前移动tileLength-lasttileLength
              DataCopy(
                  zGm[(progress - 1) * this->tileLength + this->lasttileLength],
                  zLocal, this->tileLength);
            }
          } else {
            DataCopy(zGm[progress * this->tileLength], zLocal,
                     this->tileLength);
          }
        }
        if (BUFFER_NUM == 2) {
          //开启double
          //buffer时，由于将输入数据分成了相等的2部分，分块大小为不开启double
          //buffer的一半， 所以需要对最后两个分块数据的起始地址做处理
          if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
              (progress == (this->tileNum * BUFFER_NUM - 1))) {
            //分块大小变为tileLength的一半
            //倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动
            DataCopy(
                zGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
                zLocal, (this->tileLength));
          }

          else {
            DataCopy(zGm[progress * (this->tileLength)], zLocal,
                     (this->tileLength));
          }
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formertileNum;
    uint32_t formertileLength;
    uint32_t formerlasttileLength;
    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t tailtileNum;
    uint32_t tailtileLength;
    uint32_t taillasttileLength;
};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinh op;
    uint32_t tilingKey = 1;
    if (TILING_KEY_IS(1)) {
        tilingKey = 1;
    } else if (TILING_KEY_IS(2)) {
        tilingKey = 2;
    } else {
        tilingKey = 1;
    }
    op.Init(x, z, tiling_data.blockLength,
          tiling_data.tileNum, tiling_data.tileLength,
          tiling_data.lasttileLength, tiling_data.formerNum,
          tiling_data.formerLength, tiling_data.formertileNum,
          tiling_data.formertileLength, tiling_data.formerlasttileLength,
          tiling_data.tailNum, tiling_data.tailLength, tiling_data.tailtileNum,
          tiling_data.tailtileLength, tiling_data.taillasttileLength,
          tilingKey);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void sinh_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* z,
    uint8_t* workspace, uint8_t* tiling)
{
    sinh_custom<<<blockDim, l2ctrl, stream>>>(x, z, workspace, tiling);
}
#endif
