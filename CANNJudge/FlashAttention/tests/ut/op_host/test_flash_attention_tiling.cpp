#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "flash_attention_tiling_data.h"

namespace FlashAttentionUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "FlashAttention";

struct FlashAttentionTestParam {
    std::string caseName;
    std::initializer_list<int64_t> qShape;
    ge::DataType qDtype;
    ge::Format qFormat;
    std::initializer_list<int64_t> kShape;
    ge::DataType kDtype;
    ge::Format kFormat;
    std::initializer_list<int64_t> vShape;
    ge::DataType vDtype;
    ge::Format vFormat;
    std::initializer_list<int64_t> attn_maskShape;
    ge::DataType attn_maskDtype;
    ge::Format attn_maskFormat;
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
//   expectTilingKey:  参考 op_kernel/flash_attention_tiling_key.h 和 op_host/flash_attention_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/flash_attention_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/flash_attention_tiling.cpp 中 GetWorkspaceSize 的逻辑
static FlashAttentionTestParam testCases[] = {
    {"flash_attention_0", {1, 2, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::DT_UNDEFINED, ge::FORMAT_ND, {1, 2, 8, 16}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class FlashAttentionTilingTest : public testing::TestWithParam<FlashAttentionTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "FlashAttentionTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "FlashAttentionTilingTest TearDown." << std::endl;
    }
};

struct FlashAttentionCompileInfo {} compileInfo;

static void TestOneParamCase(const FlashAttentionTestParam &param)
{
    gert::StorageShape qShape = {param.qShape, param.qShape};
    gert::StorageShape kShape = {param.kShape, param.kShape};
    gert::StorageShape vShape = {param.vShape, param.vShape};
    gert::StorageShape attn_maskShape = {param.attn_maskShape, param.attn_maskShape};
    gert::StorageShape outShape = {param.outShape, param.outShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{qShape, param.qDtype, param.qFormat},
        {kShape, param.kDtype, param.kFormat},
        {vShape, param.vDtype, param.vFormat},
        {attn_maskShape, param.attn_maskDtype, param.attn_maskFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{outShape, param.outDtype, param.outFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("softmax_scale", Ops::Math::AnyValue::CreateFrom<float>(0.0f)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("is_causal", Ops::Math::AnyValue::CreateFrom<bool>(true)));
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

TEST_P(FlashAttentionTilingTest, tiling_test)
{
    const FlashAttentionTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    FlashAttentionTilingTests,
    FlashAttentionTilingTest,
    testing::ValuesIn(testCases));

}
