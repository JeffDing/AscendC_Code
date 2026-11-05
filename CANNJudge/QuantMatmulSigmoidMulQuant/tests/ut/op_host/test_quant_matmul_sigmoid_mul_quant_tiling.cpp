#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "quant_matmul_sigmoid_mul_quant_tiling_data.h"

namespace QuantMatmulSigmoidMulQuantUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "QuantMatmulSigmoidMulQuant";

struct QuantMatmulSigmoidMulQuantTestParam {
    std::string caseName;
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;
    std::initializer_list<int64_t> x1ScaleShape;
    ge::DataType x1ScaleDtype;
    ge::Format x1ScaleFormat;
    std::initializer_list<int64_t> x2ScaleShape;
    ge::DataType x2ScaleDtype;
    ge::Format x2ScaleFormat;
    std::initializer_list<int64_t> biasShape;
    ge::DataType biasDtype;
    ge::Format biasFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> yScaleShape;
    ge::DataType yScaleDtype;
    ge::Format yScaleFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/quant_matmul_sigmoid_mul_quant_tiling_key.h 和 op_host/quant_matmul_sigmoid_mul_quant_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/quant_matmul_sigmoid_mul_quant_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/quant_matmul_sigmoid_mul_quant_tiling.cpp 中 GetWorkspaceSize 的逻辑
static QuantMatmulSigmoidMulQuantTestParam testCases[] = {
    {"quant_matmul_sigmoid_mul_quant_0", {1, 16}, ge::DT_INT8, ge::FORMAT_ND, {16, 1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::DT_UNDEFINED, ge::FORMAT_ND, {1, 1}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class QuantMatmulSigmoidMulQuantTilingTest : public testing::TestWithParam<QuantMatmulSigmoidMulQuantTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "QuantMatmulSigmoidMulQuantTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "QuantMatmulSigmoidMulQuantTilingTest TearDown." << std::endl;
    }
};

struct QuantMatmulSigmoidMulQuantCompileInfo {} compileInfo;

static void TestOneParamCase(const QuantMatmulSigmoidMulQuantTestParam &param)
{
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape x1ScaleShape = {param.x1ScaleShape, param.x1ScaleShape};
    gert::StorageShape x2ScaleShape = {param.x2ScaleShape, param.x2ScaleShape};
    gert::StorageShape biasShape = {param.biasShape, param.biasShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape yScaleShape = {param.yScaleShape, param.yScaleShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1Shape, param.x1Dtype, param.x1Format},
        {x2Shape, param.x2Dtype, param.x2Format},
        {x1ScaleShape, param.x1ScaleDtype, param.x1ScaleFormat},
        {x2ScaleShape, param.x2ScaleDtype, param.x2ScaleFormat},
        {biasShape, param.biasDtype, param.biasFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat},
        {yScaleShape, param.yScaleDtype, param.yScaleFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("transposeX1", Ops::Math::AnyValue::CreateFrom<bool>(false)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("transposeX2", Ops::Math::AnyValue::CreateFrom<bool>(false)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(QuantMatmulSigmoidMulQuantTilingTest, tiling_test)
{
    const QuantMatmulSigmoidMulQuantTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    QuantMatmulSigmoidMulQuantTilingTests,
    QuantMatmulSigmoidMulQuantTilingTest,
    testing::ValuesIn(testCases));

}
