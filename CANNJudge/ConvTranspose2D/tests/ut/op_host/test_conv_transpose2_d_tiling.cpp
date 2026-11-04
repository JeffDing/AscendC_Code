#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "conv_transpose2_d_tiling_data.h"

namespace ConvTranspose2DUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "ConvTranspose2D";

struct ConvTranspose2DTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> wShape;
    ge::DataType wDtype;
    ge::Format wFormat;
    std::initializer_list<int64_t> bShape;
    ge::DataType bDtype;
    ge::Format bFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
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
//   expectTilingKey:  参考 op_kernel/conv_transpose2_d_tiling_key.h 和 op_host/conv_transpose2_d_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/conv_transpose2_d_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/conv_transpose2_d_tiling.cpp 中 GetWorkspaceSize 的逻辑
static ConvTranspose2DTestParam testCases[] = {
    {"conv_transpose2_d_0", {1, 4, 8, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 4, 3, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4, 10, 10}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class ConvTranspose2DTilingTest : public testing::TestWithParam<ConvTranspose2DTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ConvTranspose2DTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ConvTranspose2DTilingTest TearDown." << std::endl;
    }
};

struct ConvTranspose2DCompileInfo {} compileInfo;

static void TestOneParamCase(const ConvTranspose2DTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape wShape = {param.wShape, param.wShape};
    gert::StorageShape bShape = {param.bShape, param.bShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {wShape, param.wDtype, param.wFormat},
        {bShape, param.bDtype, param.bFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("padding", Ops::Math::AnyValue::CreateFrom<int64_t>(0)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("output_padding", Ops::Math::AnyValue::CreateFrom<int64_t>(0)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("dilation", Ops::Math::AnyValue::CreateFrom<int64_t>(1)));
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

TEST_P(ConvTranspose2DTilingTest, tiling_test)
{
    const ConvTranspose2DTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ConvTranspose2DTilingTests,
    ConvTranspose2DTilingTest,
    testing::ValuesIn(testCases));

}
