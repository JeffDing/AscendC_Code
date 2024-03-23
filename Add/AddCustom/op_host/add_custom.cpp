
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
    const uint32_t BLOCK_DIM = 8;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context){
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        
        // 设置每个块的维度，设置TilingData
        context->SetBlockDim(BLOCK_DIM);
        tiling.set_totalLength(totalLength);	// 设置总计算数据量
        tiling.set_tileNum(TILE_NUM);			// 设置每个核上的tile数量
        
        // 将TilingData实例序列化并保存到TilingContext中，以便后续在kernel侧使用。
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        
        // 设置TilingKey（可选），用于选择kernel实现分支
        context->SetTilingKey(1);
        
        // 设置workspace大小（可选）
        // 如果需要在设备侧Global Memory上分配workspace内存，可以通过GetWorkspaceSizes获取大小指针并设置。
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        
        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static graphStatus InferShape(gert::InferShapeContext* context) {
        const auto inputShape = context->GetInputShape(0); // 获取输入Tensor的形状
        auto outputShape = context->GetOutputShape(0);     // 获取输出Tensor的形状
        *outputShape = *inputShape;                        // 将输入形状赋给输出形状
        return GRAPH_SUCCESS;                              // 返回成功状态
    }
} 


namespace ops {
    class AddCustom : public OpDef {
    public:
        explicit AddCustom(const char* name) : OpDef(name){
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310B4");
        }
    };
    OP_ADD(AddCustom);
}
