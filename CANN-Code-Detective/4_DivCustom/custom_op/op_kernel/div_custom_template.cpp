#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelDivCustomTemplate {
public:
    __aicore__ inline KernelDivCustomTemplate() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t blockLength, uint32_t tileLength)
    {
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        
        // 计算当前核处理的数据偏移
        uint32_t blockIdx = GetBlockIdx();
        uint32_t offset = blockIdx * blockLength;
        
        // 设置GlobalTensor，绑定GM地址
        xGm.SetGlobalBuffer((__gm__ half*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + offset, blockLength);
        
        // 初始化UB队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }
    
    __aicore__ inline void Process()
    {
        // 计算需要处理的tile数量
        uint32_t loopCount = (blockLength + tileLength - 1) / tileLength;
        
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        uint32_t currentTileLength = tileLength;
        uint32_t remainingLength = blockLength - progress * tileLength;
        if (remainingLength < tileLength) {
            currentTileLength = remainingLength;
        }
        
        // 分配UB内存
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        
        // 从GM拷贝数据到UB
        DataCopy(xLocal, xGm[progress * tileLength], currentTileLength);
        DataCopy(yLocal, yGm[progress * tileLength], currentTileLength);
        
        // 入队
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    
    __aicore__ inline void Compute(uint32_t progress)
    {
        uint32_t currentTileLength = tileLength;
        uint32_t remainingLength = blockLength - progress * tileLength;
        if (remainingLength < tileLength) {
            currentTileLength = remainingLength;
        }
        
        // 从队列中取出数据
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        
        // 执行除法计算: z = x / y
        Div(zLocal, xLocal, yLocal, currentTileLength);
        
        // 结果入队
        outQueueZ.EnQue(zLocal);
        
        // 释放输入tensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        uint32_t currentTileLength = tileLength;
        uint32_t remainingLength = blockLength - progress * tileLength;
        if (remainingLength < tileLength) {
            currentTileLength = remainingLength;
        }
        
        // 从队列中取出结果
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        
        // 从UB拷贝数据到GM
        DataCopy(zGm[progress * tileLength], zLocal, currentTileLength);
        
        // 释放输出tensor
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    KernelDivCustomTemplate op;
    op.Init(x, y, z, tiling_data.size, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
