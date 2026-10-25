#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "mhc_sinkhorn_grad_tiling_data.h"

namespace MhcSinkhornGradUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "MhcSinkhornGrad";

struct MhcSinkhornGradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> gradOutputShape;
    ge::DataType gradOutputDtype;
    ge::Format gradOutputFormat;
    std::initializer_list<int64_t> normOutShape;
    ge::DataType normOutDtype;
    ge::Format normOutFormat;
    std::initializer_list<int64_t> sumOutShape;
    ge::DataType sumOutDtype;
    ge::Format sumOutFormat;
    std::initializer_list<int64_t> gradInputShape;
    ge::DataType gradInputDtype;
    ge::Format gradInputFormat;
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
//   expectTilingKey:  参考 op_kernel/mhc_sinkhorn_grad_tiling_key.h 和 op_host/mhc_sinkhorn_grad_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/mhc_sinkhorn_grad_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/mhc_sinkhorn_grad_tiling.cpp 中 GetWorkspaceSize 的逻辑
static MhcSinkhornGradTestParam testCases[] = {
    {"mhc_sinkhorn_grad_0", {1, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 1, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 1, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class MhcSinkhornGradTilingTest : public testing::TestWithParam<MhcSinkhornGradTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "MhcSinkhornGradTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MhcSinkhornGradTilingTest TearDown." << std::endl;
    }
};

struct MhcSinkhornGradCompileInfo {} compileInfo;

static void TestOneParamCase(const MhcSinkhornGradTestParam &param)
{
    gert::StorageShape gradOutputShape = {param.gradOutputShape, param.gradOutputShape};
    gert::StorageShape normOutShape = {param.normOutShape, param.normOutShape};
    gert::StorageShape sumOutShape = {param.sumOutShape, param.sumOutShape};
    gert::StorageShape gradInputShape = {param.gradInputShape, param.gradInputShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{gradOutputShape, param.gradOutputDtype, param.gradOutputFormat},
        {normOutShape, param.normOutDtype, param.normOutFormat},
        {sumOutShape, param.sumOutDtype, param.sumOutFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{gradInputShape, param.gradInputDtype, param.gradInputFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

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

TEST_P(MhcSinkhornGradTilingTest, tiling_test)
{
    const MhcSinkhornGradTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    MhcSinkhornGradTilingTests,
    MhcSinkhornGradTilingTest,
    testing::ValuesIn(testCases));

}
