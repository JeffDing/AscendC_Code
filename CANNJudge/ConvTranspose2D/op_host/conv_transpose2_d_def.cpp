/*!
 * \file conv_transpose2_d_def.cpp
 * \brief ConvTranspose2D 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class ConvTranspose2D : public OpDef {
public:
    explicit ConvTranspose2D(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("w")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("b")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("stride")
        .AttrType(OPTIONAL)
        .Int(1);
    this->Attr("padding")
        .AttrType(OPTIONAL)
        .Int(0);
    this->Attr("output_padding")
        .AttrType(OPTIONAL)
        .Int(0);
    this->Attr("dilation")
        .AttrType(OPTIONAL)
        .Int(1);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ConvTranspose2D);
} // namespace ops
