#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "bn_training_update_tiling_data.h"

namespace BnTrainingUpdateUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "BnTrainingUpdate";

struct BnTrainingUpdateTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> sumShape;
    ge::DataType sumDtype;
    ge::Format sumFormat;
    std::initializer_list<int64_t> square_sumShape;
    ge::DataType square_sumDtype;
    ge::Format square_sumFormat;
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
//   expectTilingKey:  参考 op_kernel/bn_training_update_tiling_key.h 和 op_host/bn_training_update_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/bn_training_update_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/bn_training_update_tiling.cpp 中 GetWorkspaceSize 的逻辑
static BnTrainingUpdateTestParam testCases[] = {
    {"bn_training_update_0", {1, 4, 2, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4, 2, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class BnTrainingUpdateTilingTest : public testing::TestWithParam<BnTrainingUpdateTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "BnTrainingUpdateTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "BnTrainingUpdateTilingTest TearDown." << std::endl;
    }
};

struct BnTrainingUpdateCompileInfo {} compileInfo;

static void TestOneParamCase(const BnTrainingUpdateTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape sumShape = {param.sumShape, param.sumShape};
    gert::StorageShape square_sumShape = {param.square_sumShape, param.square_sumShape};
    gert::StorageShape scaleShape = {param.scaleShape, param.scaleShape};
    gert::StorageShape offsetShape = {param.offsetShape, param.offsetShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape save_meanShape = {param.save_meanShape, param.save_meanShape};
    gert::StorageShape save_varianceShape = {param.save_varianceShape, param.save_varianceShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {sumShape, param.sumDtype, param.sumFormat},
        {square_sumShape, param.square_sumDtype, param.square_sumFormat},
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

TEST_P(BnTrainingUpdateTilingTest, tiling_test)
{
    const BnTrainingUpdateTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    BnTrainingUpdateTilingTests,
    BnTrainingUpdateTilingTest,
    testing::ValuesIn(testCases));

}
