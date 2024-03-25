#include "kernel_operator.h"

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // 获取Host侧传入的Tiling参数
    GET_TILING_DATA(tilingData, tiling);
    // 初始化算子类
    KernelAdd op;
    // 算子类的初始化函数
    op.Init(x, z, tilingData.totalLength, tilingData.tileNum);
    if (TILING_KEY_IS(1)) {
        // 完成算子实现的核心逻辑
        op.Process();
    }
}

class KernelSinh {
public:
    __aicore__ inline Kernelsinh() {}
    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 使用获取到的TilingData计算得到singleCoreSize(每个核上总计算数据大小)、tileNum（每个核上分块个数）、singleTileLength（每个分块大小）等变量
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 获取当前核的起始索引
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + this->blockLength * GetBlockIdx(), this->blockLength);
        // 通过Pipe内存管理对象为输入输出Queue分配内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
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
    // 搬入函数，完成CopyIn阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从Queue中分配输入Tensor
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
         // 将GlobalTensor数据拷贝到LocalTensor
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        // 将LocalTesor放入VECIN（代表矢量编程中搬入数据的逻辑存放位置）的Queue中
        inQueueX.EnQue(xLocal);
    }
    // 计算函数，完成Compute阶段的处理，被核心Process函数调用
    __aicore__ inline void Compute(int32_t progress)
    {
        // 将Tensor从队列中取出，用于后续计算
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        // 从Queue中分配输出Tensor
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        Exp(xLocal, xLocal, this->tileLength);
	Reciprocal(zLocal, xLocal, this->tileLength);
	Sub(zLocal, xLocal, zLocal, this->tileLength);
	half scalar = 0.5;
	Muls(zLocal, zLocal, scalar, this->tileLength);
	// 将计算结果LocalTensor放入到VecOut的Queue中
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        // 释放输入Tensor
        inQueueX.FreeTensor(xLocal);
    }
    // 搬出函数，完成CopyOut阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从VecOut的Queue中取出输出Tensor
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        // 将输出Tensor拷贝到GlobalTensor中
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        // 将不再使用的LocalTensor释放
        outQueueZ.FreeTensor(zLocal);
    }


private:
    //Pipe内存管理对象
    TPipe pipe;
    //输入数据Queue队列管理对象，QuePosition为VECIN
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //输出数据Queue队列管理对象，QuePosition为VECOUT
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    //管理输入输出Global Memory内存地址的对象，其中xGm, yGm为输入，zGm为输出
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Z> zGm;
    // 每个核上总计算数据大小
    uint32_t blockLength;
    // 每个核上总计算数据分块个数
    uint32_t tileNum;
    // 每个分块大小
    uint32_t tileLength;
};