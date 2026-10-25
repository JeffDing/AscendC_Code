/*!
 * \file mhc_sinkhorn_grad_def.cpp
 * \brief MhcSinkhornGrad 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class MhcSinkhornGrad : public OpDef {
public:
    explicit MhcSinkhornGrad(const char* name) : OpDef(name)
    {
    this->Input("gradOutput")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("normOut")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("sumOut")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("gradInput")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MhcSinkhornGrad);
} // namespace ops
