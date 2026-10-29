/*!
 * \file sqrt_def.cpp
 * \brief Sqrt 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class Sqrt : public OpDef {
public:
    explicit Sqrt(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Sqrt);
} // namespace ops
