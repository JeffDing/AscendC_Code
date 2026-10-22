#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "multi_add_rms_norm_dynamic_quant_tiling_data.h"

namespace MultiAddRmsNormDynamicQuantUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "MultiAddRmsNormDynamicQuant";

struct MultiAddRmsNormDynamicQuantTestParam {
    std::string caseName;
    std::initializer_list<int64_t> x1aShape;
    ge::DataType x1aDtype;
    ge::Format x1aFormat;
    std::initializer_list<int64_t> x1bShape;
    ge::DataType x1bDtype;
    ge::Format x1bFormat;
    std::initializer_list<int64_t> x1cShape;
    ge::DataType x1cDtype;
    ge::Format x1cFormat;
    std::initializer_list<int64_t> x1dShape;
    ge::DataType x1dDtype;
    ge::Format x1dFormat;
    std::initializer_list<int64_t> x1eShape;
    ge::DataType x1eDtype;
    ge::Format x1eFormat;
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;
    std::initializer_list<int64_t> gammaShape;
    ge::DataType gammaDtype;
    ge::Format gammaFormat;
    std::initializer_list<int64_t> smoothScale1Shape;
    ge::DataType smoothScale1Dtype;
    ge::Format smoothScale1Format;
    std::initializer_list<int64_t> smoothScale2Shape;
    ge::DataType smoothScale2Dtype;
    ge::Format smoothScale2Format;
    std::initializer_list<int64_t> y1Shape;
    ge::DataType y1Dtype;
    ge::Format y1Format;
    std::initializer_list<int64_t> y2Shape;
    ge::DataType y2Dtype;
    ge::Format y2Format;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> scale1Shape;
    ge::DataType scale1Dtype;
    ge::Format scale1Format;
    std::initializer_list<int64_t> scale2Shape;
    ge::DataType scale2Dtype;
    ge::Format scale2Format;
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
//   expectTilingKey:  参考 op_kernel/multi_add_rms_norm_dynamic_quant_tiling_key.h 和 op_host/multi_add_rms_norm_dynamic_quant_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/multi_add_rms_norm_dynamic_quant_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/multi_add_rms_norm_dynamic_quant_tiling.cpp 中 GetWorkspaceSize 的逻辑
static MultiAddRmsNormDynamicQuantTestParam testCases[] = {
    {"multi_add_rms_norm_dynamic_quant_0", {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 64}, ge::DT_INT8, ge::FORMAT_ND, {1, 64}, ge::DT_INT8, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class MultiAddRmsNormDynamicQuantTilingTest : public testing::TestWithParam<MultiAddRmsNormDynamicQuantTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "MultiAddRmsNormDynamicQuantTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MultiAddRmsNormDynamicQuantTilingTest TearDown." << std::endl;
    }
};

struct MultiAddRmsNormDynamicQuantCompileInfo {} compileInfo;

static void TestOneParamCase(const MultiAddRmsNormDynamicQuantTestParam &param)
{
    gert::StorageShape x1aShape = {param.x1aShape, param.x1aShape};
    gert::StorageShape x1bShape = {param.x1bShape, param.x1bShape};
    gert::StorageShape x1cShape = {param.x1cShape, param.x1cShape};
    gert::StorageShape x1dShape = {param.x1dShape, param.x1dShape};
    gert::StorageShape x1eShape = {param.x1eShape, param.x1eShape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape gammaShape = {param.gammaShape, param.gammaShape};
    gert::StorageShape smoothScale1Shape = {param.smoothScale1Shape, param.smoothScale1Shape};
    gert::StorageShape smoothScale2Shape = {param.smoothScale2Shape, param.smoothScale2Shape};
    gert::StorageShape y1Shape = {param.y1Shape, param.y1Shape};
    gert::StorageShape y2Shape = {param.y2Shape, param.y2Shape};
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape scale1Shape = {param.scale1Shape, param.scale1Shape};
    gert::StorageShape scale2Shape = {param.scale2Shape, param.scale2Shape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1aShape, param.x1aDtype, param.x1aFormat},
        {x1bShape, param.x1bDtype, param.x1bFormat},
        {x1cShape, param.x1cDtype, param.x1cFormat},
        {x1dShape, param.x1dDtype, param.x1dFormat},
        {x1eShape, param.x1eDtype, param.x1eFormat},
        {x2Shape, param.x2Dtype, param.x2Format},
        {gammaShape, param.gammaDtype, param.gammaFormat},
        {smoothScale1Shape, param.smoothScale1Dtype, param.smoothScale1Format},
        {smoothScale2Shape, param.smoothScale2Dtype, param.smoothScale2Format}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{y1Shape, param.y1Dtype, param.y1Format},
        {y2Shape, param.y2Dtype, param.y2Format},
        {xShape, param.xDtype, param.xFormat},
        {yShape, param.yDtype, param.yFormat},
        {scale1Shape, param.scale1Dtype, param.scale1Format},
        {scale2Shape, param.scale2Dtype, param.scale2Format}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(1e-06f)));
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

TEST_P(MultiAddRmsNormDynamicQuantTilingTest, tiling_test)
{
    const MultiAddRmsNormDynamicQuantTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    MultiAddRmsNormDynamicQuantTilingTests,
    MultiAddRmsNormDynamicQuantTilingTest,
    testing::ValuesIn(testCases));

}
