#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "batch_norm3_d_tiling_data.h"

namespace BatchNorm3DUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "BatchNorm3D";

struct BatchNorm3DTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> scaleShape;
    ge::DataType scaleDtype;
    ge::Format scaleFormat;
    std::initializer_list<int64_t> offsetShape;
    ge::DataType offsetDtype;
    ge::Format offsetFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> save_meanShape;
    ge::DataType save_meanDtype;
    ge::Format save_meanFormat;
    std::initializer_list<int64_t> save_varianceShape;
    ge::DataType save_varianceDtype;
    ge::Format save_varianceFormat;
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
//   expectTilingKey:  参考 op_kernel/batch_norm3_d_tiling_key.h 和 op_host/batch_norm3_d_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/batch_norm3_d_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/batch_norm3_d_tiling.cpp 中 GetWorkspaceSize 的逻辑
static BatchNorm3DTestParam testCases[] = {
    {"batch_norm3_d_0", {1, 4, 2, 2, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4, 2, 2, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class BatchNorm3DTilingTest : public testing::TestWithParam<BatchNorm3DTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "BatchNorm3DTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "BatchNorm3DTilingTest TearDown." << std::endl;
    }
};

struct BatchNorm3DCompileInfo {} compileInfo;

static void TestOneParamCase(const BatchNorm3DTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape scaleShape = {param.scaleShape, param.scaleShape};
    gert::StorageShape offsetShape = {param.offsetShape, param.offsetShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape save_meanShape = {param.save_meanShape, param.save_meanShape};
    gert::StorageShape save_varianceShape = {param.save_varianceShape, param.save_varianceShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {scaleShape, param.scaleDtype, param.scaleFormat},
        {offsetShape, param.offsetDtype, param.offsetFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat},
        {save_meanShape, param.save_meanDtype, param.save_meanFormat},
        {save_varianceShape, param.save_varianceDtype, param.save_varianceFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(1e-05f)));
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

TEST_P(BatchNorm3DTilingTest, tiling_test)
{
    const BatchNorm3DTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    BatchNorm3DTilingTests,
    BatchNorm3DTilingTest,
    testing::ValuesIn(testCases));

}
