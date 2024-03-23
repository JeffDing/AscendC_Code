/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // seperate to 2 parts, due to double buffer

class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
	ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
	this->blockLength = totalLength/ GetBlockNum();
	this->tileNum = tileNum;
	ASSERT(tileNum != 0 && "tilen num can not be zero!");
	this->tileLength = this->blockLength /tileNum .BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half*)x + this->BLOCK_LENGTH * GetBlockIdx(), this->BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half*)z + this->BLOCK_LENGTH * GetBlockIdx(), this->BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->TILE_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->TILE_LENGTH], this->TILE_LENGTH);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Exp(xLocal, xLocal, this->TILE_LENGTH);
        Reciprocal(zLocal, xLocal, this->TILE_LENGTH);
	Sub(zLocal, xLocal, zLocal, this->TILE_LENGTH);
	half scalar = 0.5;
	Muls(zLocal, zLocal, scalar, this->TILE_LENGTH);
	outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->TILE_LENGTH], zLocal, this->TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> zGm;
};


extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd op;
    op.Init(x, z);
    op.Process();
}
