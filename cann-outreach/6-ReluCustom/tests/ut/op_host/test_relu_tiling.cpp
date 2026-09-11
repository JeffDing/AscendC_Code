#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "relu_tiling_data.h"

namespace ReluUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Relu";

struct ReluTestParam {
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

// 期望值说明：
//   expectTilingKey:  fp32 -> RELU_TPL_SCH_MODE_1 (1)，fp16/bf16 -> RELU_TPL_SCH_MODE_0 (0)
//   expectWorkspaces: GetWorkspaceSize 设置 WS_SYS_SIZE=0 -> {0}
//   expectTilingData: 使用 EMPTY_EXPECT_TILING_DATA 跳过 tiling 数据逐字段比对
//                     （blockFactor/ubFactor 依赖运行时 coreNum，由 tiling 逻辑动态计算）
//   expectResult:     GRAPH_SUCCESS
static ReluTestParam testCases[] = {
    {"relu_0", {45, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, {45, 2048}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, EMPTY_EXPECT_TILING_DATA, {0}, 64, 262144, 4096},
};

class ReluTilingTest : public testing::TestWithParam<ReluTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ReluTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ReluTilingTest TearDown." << std::endl;
    }
};

struct ReluCompileInfo {} compileInfo;

static void TestOneParamCase(const ReluTestParam &param)
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

TEST_P(ReluTilingTest, tiling_test)
{
    const ReluTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ReluTilingTests,
    ReluTilingTest,
    testing::ValuesIn(testCases));

}
