/*!
 * \file quant_matmul_sigmoid_mul_quant_def.cpp
 * \brief QuantMatmulSigmoidMulQuant 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class QuantMatmulSigmoidMulQuant : public OpDef {
public:
    explicit QuantMatmulSigmoidMulQuant(const char* name) : OpDef(name)
    {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x1Scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x2Scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("yScale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("transposeX1")
        .AttrType(OPTIONAL)
        .Bool(0);
    this->Attr("transposeX2")
        .AttrType(OPTIONAL)
        .Bool(0);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(QuantMatmulSigmoidMulQuant);
} // namespace ops
