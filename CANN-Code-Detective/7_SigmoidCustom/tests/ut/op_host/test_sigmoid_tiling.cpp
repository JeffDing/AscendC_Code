#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "sigmoid_tiling_data.h"

namespace SigmoidUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Sigmoid";

struct SigmoidTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
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
//   expectTilingKey:  参考 op_kernel/sigmoid_tiling_key.h 和 op_host/sigmoid_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/sigmoid_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/sigmoid_tiling.cpp 中 GetWorkspaceSize 的逻辑
static SigmoidTestParam testCases[] = {
    {"sigmoid_0", {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "16384 256 256 ", {0}, 64, 262144, 4096},
};

class SigmoidTilingTest : public testing::TestWithParam<SigmoidTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "SigmoidTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "SigmoidTilingTest TearDown." << std::endl;
    }
};

struct SigmoidCompileInfo {} compileInfo;

static void TestOneParamCase(const SigmoidTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
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

TEST_P(SigmoidTilingTest, tiling_test)
{
    const SigmoidTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    SigmoidTilingTests,
    SigmoidTilingTest,
    testing::ValuesIn(testCases));

}
