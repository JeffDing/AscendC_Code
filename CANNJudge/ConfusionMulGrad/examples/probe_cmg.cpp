#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_confusion_mul_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

static std::vector<uint8_t> ReadFile(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "open %s failed\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (sz > 0 && fread(buf.data(), 1, sz, f) != (size_t)sz) {
        fclose(f);
        fprintf(stderr, "read %s failed\n", path);
        exit(1);
    }
    fclose(f);
    return buf;
}

static void WriteFile(const char* path, const void* data, size_t sz)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "open %s for write failed\n", path);
        exit(1);
    }
    if (sz > 0) {
        fwrite(data, 1, sz, f);
    }
    fclose(f);
}

static void* Upload(const std::vector<uint8_t>& host)
{
    void* dev = nullptr;
    auto ret = aclrtMalloc(&dev, host.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtMalloc failed. ERROR: %d\n", ret); exit(1));
    if (!host.empty()) {
        ret = aclrtMemcpy(dev, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, printf("aclrtMemcpy failed. ERROR: %d\n", ret); exit(1));
    }
    return dev;
}

// argc: prog inPrefix M N axisCode(1=row/0=col/2=all) dtype(0=f32,1=f16) outPrefix
int main(int argc, char** argv)
{
    if (argc < 7) {
        printf("usage: %s inPrefix M N axisCode dtype outPrefix\n", argv[0]);
        return 1;
    }
    const char* inPrefix = argv[1];
    const int64_t M = atoll(argv[2]);
    const int64_t N = atoll(argv[3]);
    const int axisCode = atoi(argv[4]);
    const int dtypeSel = atoi(argv[5]);
    const char* outPrefix = argv[6];

    const aclDataType inType = dtypeSel == 0 ? aclDataType::ACL_FLOAT : aclDataType::ACL_FLOAT16;
    const size_t esz = dtypeSel == 0 ? 4 : 2;
    const int64_t elem = M * N;
    const int64_t out1Cnt = axisCode == 0 ? N : (axisCode == 1 ? M : 1);

    char path[512];
    snprintf(path, sizeof(path), "%s_input0.bin", inPrefix);
    std::vector<uint8_t> in0 = ReadFile(path);
    snprintf(path, sizeof(path), "%s_input1.bin", inPrefix);
    std::vector<uint8_t> in1 = ReadFile(path);
    snprintf(path, sizeof(path), "%s_input2.bin", inPrefix);
    std::vector<uint8_t> in2 = ReadFile(path);
    if ((int64_t)in0.size() != elem * (int64_t)esz ||
        (int64_t)in1.size() != elem * (int64_t)esz ||
        (int64_t)in2.size() != elem * (int64_t)esz) {
        printf("input size mismatch: got %ld/%ld/%ld bytes, expect %ld\n",
               (long)in0.size(), (long)in1.size(), (long)in2.size(), (long)(elem * (int64_t)esz));
        return 1;
    }

    aclrtStream stream;
    auto ret = InitAcl(0, &stream);
    CHECK_RET(ret == 0, return ret);

    void* d0 = Upload(in0);
    void* d1 = Upload(in1);
    void* d2 = Upload(in2);
    void* o0 = nullptr;
    void* o1 = nullptr;
    ret = aclrtMalloc(&o0, elem * esz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, printf("malloc o0 failed\n"); return ret);
    ret = aclrtMalloc(&o1, out1Cnt * esz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, printf("malloc o1 failed\n"); return ret);

    const int64_t inShape[2] = {M, N};
    const int64_t inStrides[2] = {N, 1};
    aclTensor* t0 = aclCreateTensor(inShape, 2, inType, inStrides, 0, aclFormat::ACL_FORMAT_ND,
                                    inShape, 2, d0);
    aclTensor* t1 = aclCreateTensor(inShape, 2, inType, inStrides, 0, aclFormat::ACL_FORMAT_ND,
                                    inShape, 2, d1);
    aclTensor* t2 = aclCreateTensor(inShape, 2, inType, inStrides, 0, aclFormat::ACL_FORMAT_ND,
                                    inShape, 2, d2);
    const int64_t out0Shape[2] = {M, N};
    const int64_t out0Strides[2] = {N, 1};
    aclTensor* to0 = aclCreateTensor(out0Shape, 2, inType, out0Strides, 0, aclFormat::ACL_FORMAT_ND,
                                     out0Shape, 2, o0);
    const int64_t out1Shape[1] = {out1Cnt};
    const int64_t out1Strides[1] = {1};
    aclTensor* to1 = aclCreateTensor(out1Shape, 1, inType, out1Strides, 0, aclFormat::ACL_FORMAT_ND,
                                     out1Shape, 1, o1);

    int64_t axisData[2] = {0, 1};
    int axisLen = 1;
    if (axisCode == 1) {
        axisData[0] = 1;
    } else if (axisCode == 2) {
        axisLen = 2;
    } else {
        axisData[0] = 0;
    }
    aclIntArray* axis = aclCreateIntArray(axisData, axisLen);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnConfusionMulGradGetWorkspaceSize(t0, t1, t2, axis, false, to0, to1, &workspaceSize, &executor);
    CHECK_RET(ret == 0, printf("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, printf("malloc ws failed\n"); return ret);
    }
    ret = aclnnConfusionMulGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, printf("aclnnConfusionMulGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("Sync failed. ERROR: %d\n", ret); return ret);

    // 性能迭代(不含首次预热): argv[7] = iters, 默认 0(不测)
    const int iters = argc > 7 ? atoi(argv[7]) : 0;
    if (iters > 0) {
        auto c0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            uint64_t ws2 = 0;
            aclOpExecutor* ex2 = nullptr;
            auto r2 = aclnnConfusionMulGradGetWorkspaceSize(t0, t1, t2, axis, false, to0, to1, &ws2, &ex2);
            CHECK_RET(r2 == 0, printf("GetWorkspaceSize(%d) failed\n", it); return r2);
            r2 = aclnnConfusionMulGrad(workspaceAddr, ws2 > 0 ? workspaceSize : 0, ex2, stream);
            CHECK_RET(r2 == 0, printf("aclnn(%d) failed\n", it); return r2);
        }
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("Sync failed. ERROR: %d\n", ret); return ret);
        auto c1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(c1 - c0).count() / iters;
        printf("perf: %.2f us/iter (iters=%d)\n", us, iters);
    }

    std::vector<uint8_t> hostO0(elem * esz);
    std::vector<uint8_t> hostO1(out1Cnt * esz);
    ret = aclrtMemcpy(hostO0.data(), hostO0.size(), o0, elem * esz, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, printf("memcpy o0 failed\n"); return ret);
    ret = aclrtMemcpy(hostO1.data(), hostO1.size(), o1, out1Cnt * esz, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, printf("memcpy o1 failed\n"); return ret);

    snprintf(path, sizeof(path), "%s_out0.bin", outPrefix);
    WriteFile(path, hostO0.data(), hostO0.size());
    snprintf(path, sizeof(path), "%s_out1.bin", outPrefix);
    WriteFile(path, hostO1.data(), hostO1.size());

    aclDestroyTensor(t0);
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(to0);
    aclDestroyTensor(to1);
    aclDestroyIntArray(axis);
    aclrtFree(d0);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(o0);
    aclrtFree(o1);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    printf("probe done: M=%ld N=%ld axis=%d dtype=%s out1Cnt=%ld\n", (long)M, (long)N, axisCode,
           dtypeSel == 0 ? "fp32" : "fp16", (long)out1Cnt);
    return 0;
}
