#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "mse_loss_tiling_data.h"

namespace MseLossUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "MseLoss";

struct MseLossTestParam {
    std::string caseName;
    std::initializer_list<int64_t> selfShape;
    ge::DataType selfDtype;
    ge::Format selfFormat;
    std::initializer_list<int64_t> targetShape;
    ge::DataType targetDtype;
    ge::Format targetFormat;
    std::initializer_list<int64_t> outShape;
    ge::DataType outDtype;
    ge::Format outFormat;
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
//   expectTilingKey:  参考 op_kernel/mse_loss_tiling_key.h 和 op_host/mse_loss_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/mse_loss_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/mse_loss_tiling.cpp 中 GetWorkspaceSize 的逻辑
static MseLossTestParam testCases[] = {
    {"mse_loss_0", {8}, ge::DT_FLOAT, ge::FORMAT_ND, {8}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class MseLossTilingTest : public testing::TestWithParam<MseLossTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "MseLossTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MseLossTilingTest TearDown." << std::endl;
    }
};

struct MseLossCompileInfo {} compileInfo;

static void TestOneParamCase(const MseLossTestParam &param)
{
    gert::StorageShape selfShape = {param.selfShape, param.selfShape};
    gert::StorageShape targetShape = {param.targetShape, param.targetShape};
    gert::StorageShape outShape = {param.outShape, param.outShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{selfShape, param.selfDtype, param.selfFormat},
        {targetShape, param.targetDtype, param.targetFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{outShape, param.outDtype, param.outFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("reduction", Ops::Math::AnyValue::CreateFrom<int64_t>(1)));
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

TEST_P(MseLossTilingTest, tiling_test)
{
    const MseLossTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    MseLossTilingTests,
    MseLossTilingTest,
    testing::ValuesIn(testCases));

}
