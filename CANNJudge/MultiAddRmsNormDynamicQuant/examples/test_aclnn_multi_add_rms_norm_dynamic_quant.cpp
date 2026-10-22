#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <ctime>
#include "acl/acl.h"
#include "aclnn_multi_add_rms_norm_dynamic_quant.h"

#define CHECK_RET(cond, return_expr) do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) { int64_t s=1; for(auto v:shape) s*=v; return s; }

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed %d\n", ret); return ret);
    return 0;
}

// 直接按位传递（fp16/int8/fp32 均不转换），避免 FloatToHalf 截断精度损失
int CreateTensorRaw(const std::vector<uint8_t>& rawBytes, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = 4;
    if (dataType == aclDataType::ACL_FLOAT16 || dataType == aclDataType::ACL_BF16) elemSize = 2;
    else if (dataType == aclDataType::ACL_INT8 || dataType == aclDataType::ACL_UINT8) elemSize = 1;
    int64_t size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, rawBytes.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) strides[i] = shape[i+1] * strides[i+1];
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

std::vector<uint8_t> ReadBin(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { LOG_PRINT("cannot open %s\n", path); return {}; }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int WriteBin(const char* path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return -1;
    f.write((const char*)data, n);
    return 0;
}

int main(int argc, char** argv)
{
    const char* DATA = (argc > 1) ? argv[1] : "/tmp/repro_data";
    const int64_t B = (argc > 2) ? atoll(argv[2]) : 1024;
    const int64_t H = (argc > 3) ? atoll(argv[3]) : 8192;
    const bool isBf16 = (argc > 4) && (argv[4][0] == 'b');
    const aclDataType inDt = isBf16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
    std::vector<int64_t> xShape = {B, H};
    std::vector<int64_t> wShape = {H};
    std::vector<int64_t> scShape = {B};

    int32_t deviceId = 0; aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto mk = [&](const char* name, std::vector<int64_t> shape, aclDataType dt, void** addr, aclTensor** t) {
        auto raw = ReadBin((std::string(DATA) + "/in_" + name + ".bin").c_str());
        CHECK_RET(!raw.empty(), LOG_PRINT("read %s empty\n", name); return -1);
        return CreateTensorRaw(raw, shape, addr, dt, t);
    };

    aclTensor *x1a=nullptr,*x1b=nullptr,*x1c=nullptr,*x1d=nullptr,*x1e=nullptr,*x2=nullptr;
    aclTensor *gamma=nullptr,*sm1=nullptr,*sm2=nullptr;
    void *ax1a=nullptr,*ax1b=nullptr,*ax1c=nullptr,*ax1d=nullptr,*ax1e=nullptr,*ax2=nullptr;
    void *ag=nullptr,*as1=nullptr,*as2=nullptr;
    CHECK_RET(mk("x1a",xShape,inDt,&ax1a,&x1a)==0, return -1);
    CHECK_RET(mk("x1b",xShape,inDt,&ax1b,&x1b)==0, return -1);
    CHECK_RET(mk("x1c",xShape,inDt,&ax1c,&x1c)==0, return -1);
    CHECK_RET(mk("x1d",xShape,inDt,&ax1d,&x1d)==0, return -1);
    CHECK_RET(mk("x1e",xShape,inDt,&ax1e,&x1e)==0, return -1);
    CHECK_RET(mk("x2",xShape,inDt,&ax2,&x2)==0, return -1);
    CHECK_RET(mk("gamma",wShape,inDt,&ag,&gamma)==0, return -1);
    CHECK_RET(mk("smoothScale1",wShape,inDt,&as1,&sm1)==0, return -1);
    CHECK_RET(mk("smoothScale2",wShape,inDt,&as2,&sm2)==0, return -1);

    // 输出
    aclTensor *y1=nullptr,*y2=nullptr,*xo=nullptr,*yo=nullptr,*sc1=nullptr,*sc2=nullptr;
    void *ay1=nullptr,*ay2=nullptr,*axo=nullptr,*ayo=nullptr,*asc1=nullptr,*asc2=nullptr;
    std::vector<uint8_t> zeroBH(B*H*2, 0), zeroSc(B*4, 0), zeroI8(B*H, 0);
    CHECK_RET(CreateTensorRaw(zeroI8, xShape, &ay1, ACL_INT8, &y1)==0, return -1);
    CHECK_RET(CreateTensorRaw(zeroI8, xShape, &ay2, ACL_INT8, &y2)==0, return -1);
    CHECK_RET(CreateTensorRaw(zeroBH, xShape, &axo, ACL_FLOAT16, &xo)==0, return -1);
    CHECK_RET(CreateTensorRaw(zeroBH, xShape, &ayo, ACL_FLOAT16, &yo)==0, return -1);
    CHECK_RET(CreateTensorRaw(zeroSc, scShape, &asc1, ACL_FLOAT, &sc1)==0, return -1);
    CHECK_RET(CreateTensorRaw(zeroSc, scShape, &asc2, ACL_FLOAT, &sc2)==0, return -1);

    uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
    ret = aclnnMultiAddRmsNormDynamicQuantGetWorkspaceSize(x1a,x1b,x1c,x1d,x1e,x2,gamma,sm1,sm2,1e-6f,
            y1,y2,xo,yo,sc1,sc2,&workspaceSize,&executor);
    CHECK_RET(ret == 0, LOG_PRINT("GetWorkspaceSize failed %d\n", ret); return ret);
    void* ws=nullptr;
    if (workspaceSize > 0) { ret = aclrtMalloc(&ws, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST); CHECK_RET(ret==0, return ret); }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ret = aclnnMultiAddRmsNormDynamicQuant(ws, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, LOG_PRINT("aclnn failed %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    LOG_PRINT("kernel time = %.2f us\n", (t1.tv_sec-t0.tv_sec)*1e6 + (t1.tv_nsec-t0.tv_nsec)/1e3);
    CHECK_RET(ret == 0, LOG_PRINT("Sync failed %d\n", ret); return ret);

    // 读回输出
    std::vector<int8_t> hY1(B*H), hY2(B*H);
    std::vector<float> hSc1(B), hSc2(B);
    aclrtMemcpy(hY1.data(), B*H, ay1, B*H, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hY2.data(), B*H, ay2, B*H, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hSc1.data(), B*4, asc1, B*4, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hSc2.data(), B*4, asc2, B*4, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteBin((std::string(DATA)+"/out_y1.bin").c_str(), hY1.data(), B*H);
    WriteBin((std::string(DATA)+"/out_y2.bin").c_str(), hY2.data(), B*H);
    WriteBin((std::string(DATA)+"/out_scale1.bin").c_str(), hSc1.data(), B*4);
    WriteBin((std::string(DATA)+"/out_scale2.bin").c_str(), hSc2.data(), B*4);
    std::vector<uint8_t> hX(B*H*2), hY(B*H*2);
    aclrtMemcpy(hX.data(), B*H*2, axo, B*H*2, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hY.data(), B*H*2, ayo, B*H*2, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteBin((std::string(DATA)+"/out_x.bin").c_str(), hX.data(), B*H*2);
    WriteBin((std::string(DATA)+"/out_y.bin").c_str(), hY.data(), B*H*2);
    LOG_PRINT("NPU run done. B=%ld H=%ld\n", B, H);

    aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
    return 0;
}
