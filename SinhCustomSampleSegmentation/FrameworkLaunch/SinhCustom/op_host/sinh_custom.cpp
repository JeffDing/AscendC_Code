
#include "sinh_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  SinhCustomTilingData tiling;
  uint32_t sizeofdatatype;
  uint32_t totalLengthAligned;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto socVersion = ascendcPlatform.GetSocVersion();
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  auto aicNum = ascendcPlatform.GetCoreNumAic();

  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
  auto dt = context->GetInputTensor(0)->GetDataType();
  if (dt == 1) {
    sizeofdatatype = 2;
  }

  uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
  //核内拆分，策略是尽可能的填满ub_size，最后一包单独处理，
  uint32_t ub_block_num_real =
      ((ub_size) / BLOCK_SIZE / 2) * 13 /
      20;  // ub_block在Ascend C中不能全部被用来作为输入输出，给了13/20系数。
  uint32_t ub_block_num = 5;  //为测试方便，验证代码流程
  uint32_t tile_num;

  if (ub_block_num % 2 != 0) {
    ub_block_num = ub_block_num - 1;
  }

  // get attr
  const float* pvalue = context->GetAttrs()->GetFloat(0);
  tiling.set_value(*pvalue);

  // 1.输入向量满足32字节对齐
  if (totalLength % ALIGN_NUM != 0) {  //不对齐，先32位对齐
    totalLengthAligned =
        ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
  } else {
    totalLengthAligned = totalLength;
  }

  if (totalLengthAligned <= ub_block_num * ALIGN_NUM) {  // shape较小，用单核
    context->SetBlockDim(1);
  } else {
    if (((totalLengthAligned / ALIGN_NUM) % ub_block_num) ==
        0) {  //可以核间均分
      if ((totalLengthAligned / ALIGN_NUM / ub_block_num) <=
          aivNum) {  //且计算出均分后的核数小于当前aicore数量，按计算值
        context->SetBlockDim(totalLengthAligned / ALIGN_NUM / ub_block_num);

      } else {
        // ... 按照aivNum切分
        // context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum,
        // aivNum));
        context->SetBlockDim(aivNum);
      }
    } else {  //核间不能均分
      if (((totalLengthAligned / ALIGN_NUM / ub_block_num) + 1) <=
          aivNum) {  //且计算出均分后的核数小于当前aicore数量，按计算值
        context->SetBlockDim((totalLengthAligned / ALIGN_NUM / ub_block_num) +
                             1);
      } else {
        // ... 按照aivNum切分
        // context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum,
        // aivNum));
        context->SetBlockDim(aivNum);
      }
    }
  }

  auto block_dim = context->GetBlockDim();
  uint32_t blockLength = 0;
  uint32_t tileLength = 0;
  uint32_t lasttileLength = 0;

  uint32_t formertileLength = 0;
  uint32_t formerlasttileLength = 0;
  if ((totalLengthAligned / ALIGN_NUM) % block_dim == 0) {  //核间可均分
    blockLength = totalLengthAligned / block_dim;
    tile_num = blockLength / ALIGN_NUM / ub_block_num;
    if ((totalLengthAligned / block_dim / ALIGN_NUM) % ub_block_num == 0 ||
        tile_num == 0) {  //满足32字节对齐，可以核内均分
      if (tile_num == 0) {
        tile_num = 1;
      } 
      if (blockLength < ub_block_num) {
        tileLength = ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
        lasttileLength = tileLength;
      } else {
        tileLength = ub_block_num * ALIGN_NUM;
        lasttileLength = tileLength;
      }
    } else {  //满足32字节对齐，核内不能均分
      tile_num = tile_num + 1;
      tileLength = ub_block_num * ALIGN_NUM;
      lasttileLength = blockLength - (tile_num - 1) * tileLength;
    }
    context->SetTilingKey(1);
    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tileLength);
    tiling.set_lasttileLength(lasttileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
  } else {
    uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % block_dim;
    uint32_t tailNum = block_dim - formerNum;
    // 计算大块和小块的数据量
    uint32_t formerLength =
        (((totalLengthAligned + block_dim - 1) / block_dim + ALIGN_NUM - 1) /
         ALIGN_NUM) *
        ALIGN_NUM;
    uint32_t tailLength =
        (totalLengthAligned / block_dim / ALIGN_NUM) * ALIGN_NUM;

    bool isformershare = true;
    uint32_t former_tile_num = formerLength / ALIGN_NUM / ub_block_num;
    if ((formerLength / ALIGN_NUM) % ub_block_num == 0 ||
        former_tile_num == 0) {  //核内均分
      if (former_tile_num == 0) {
        former_tile_num = 1;
      }
      if (formerLength < ub_block_num * ALIGN_NUM) {
        formertileLength = ((formerLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
        formerlasttileLength = formertileLength;
      } else {
        formertileLength = ub_block_num * ALIGN_NUM;
        formerlasttileLength = formertileLength;
      }
    } else {
      isformershare = false;
      former_tile_num = former_tile_num + 1;

      formertileLength = ub_block_num * ALIGN_NUM;
      formerlasttileLength =
          (formerLength - (former_tile_num - 1) * formertileLength);
    }

    bool istailshare = true;
    uint32_t tail_tile_num = tailLength / ALIGN_NUM / ub_block_num;
    uint32_t tailtileLength;
    uint32_t taillasttileLength;
    if ((tailLength / ALIGN_NUM) % ub_block_num == 0 ||
        tail_tile_num == 0) {  //核内可以均分
      if (tail_tile_num == 0) {
        tail_tile_num = 1;
      }
      if (tailLength < (ub_block_num * ALIGN_NUM)) {
        tailtileLength = ((tailLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
        taillasttileLength = tailtileLength;
      } else {
        tailtileLength = ub_block_num * ALIGN_NUM;
        taillasttileLength = tailtileLength;
      }
    } else {  //核内不均分
      istailshare = false;
      tail_tile_num = tail_tile_num + 1;
      tailtileLength = ub_block_num * ALIGN_NUM;
      taillasttileLength = (tailLength - (tail_tile_num - 1) * tailtileLength);
    }
    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_formertileNum(former_tile_num);
    tiling.set_formertileLength(formertileLength);
    tiling.set_formerlasttileLength(formerlasttileLength);
    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);
    tiling.set_tailtileNum(tail_tile_num);
    tiling.set_tailtileLength(tailtileLength);
    tiling.set_taillasttileLength(taillasttileLength);
    context->SetTilingKey(2);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
  }
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SinhCustom : public OpDef {
 public:
  explicit SinhCustom(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("value").AttrType(OPTIONAL).Float(1.0);

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910");
    this->AICore().AddConfig("ascend310b");
  }
};

OP_ADD(SinhCustom);
}  // namespace ops
