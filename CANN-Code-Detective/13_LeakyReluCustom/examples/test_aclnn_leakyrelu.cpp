#include "sample_support.h"
#include "aclnn_leakyrelu.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({8, 2048}, ACL_FLOAT, 65536, "x.bin", true, false, root);
            Tensor output_0_0({8, 2048}, ACL_FLOAT, 65536, "y.bin", false, false, root);
            double attr_0 = static_cast<double>(0.01);
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnLeakyreluGetWorkspaceSize(
                input_0_0.tensor,
                attr_0,
                output_0_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnLeakyrelu(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            if (workspace)
                check(aclrtFree(workspace));
        }
        check(aclrtDestroyStream(stream));
        check(aclrtResetDevice(device));
        check(aclFinalize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
