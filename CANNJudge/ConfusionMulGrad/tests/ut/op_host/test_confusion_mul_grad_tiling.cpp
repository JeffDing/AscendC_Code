#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "confusion_mul_grad_tiling_data.h"

namespace ConfusionMulGradUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "ConfusionMulGrad";

struct ConfusionMulGradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> input0Shape;
    ge::DataType input0Dtype;
    ge::Format input0Format;
    std::initializer_list<int64_t> input1Shape;
    ge::DataType input1Dtype;
    ge::Format input1Format;
    std::initializer_list<int64_t> input2Shape;
    ge::DataType input2Dtype;
    ge::Format input2Format;
    std::initializer_list<int64_t> output0Shape;
    ge::DataType output0Dtype;
    ge::Format output0Format;
    std::initializer_list<int64_t> output1Shape;
    ge::DataType output1Dtype;
    ge::Format output1Format;
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
//   expectTilingKey:  参考 op_kernel/confusion_mul_grad_tiling_key.h 和 op_host/confusion_mul_grad_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/confusion_mul_grad_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/confusion_mul_grad_tiling.cpp 中 GetWorkspaceSize 的逻辑
static ConfusionMulGradTestParam testCases[] = {
    {"confusion_mul_grad_0", {3, 64}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 64}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 64}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 64}, ge::DT_FLOAT, ge::FORMAT_ND, {3}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class ConfusionMulGradTilingTest : public testing::TestWithParam<ConfusionMulGradTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ConfusionMulGradTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ConfusionMulGradTilingTest TearDown." << std::endl;
    }
};

struct ConfusionMulGradCompileInfo {} compileInfo;

static void TestOneParamCase(const ConfusionMulGradTestParam &param)
{
    gert::StorageShape input0Shape = {param.input0Shape, param.input0Shape};
    gert::StorageShape input1Shape = {param.input1Shape, param.input1Shape};
    gert::StorageShape input2Shape = {param.input2Shape, param.input2Shape};
    gert::StorageShape output0Shape = {param.output0Shape, param.output0Shape};
    gert::StorageShape output1Shape = {param.output1Shape, param.output1Shape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{input0Shape, param.input0Dtype, param.input0Format},
        {input1Shape, param.input1Dtype, param.input1Format},
        {input2Shape, param.input2Dtype, param.input2Format}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{output0Shape, param.output0Dtype, param.output0Format},
        {output1Shape, param.output1Dtype, param.output1Format}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})));
    attrs_.push_back(gert::TilingContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)));
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

TEST_P(ConfusionMulGradTilingTest, tiling_test)
{
    const ConfusionMulGradTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ConfusionMulGradTilingTests,
    ConfusionMulGradTilingTest,
    testing::ValuesIn(testCases));

}
