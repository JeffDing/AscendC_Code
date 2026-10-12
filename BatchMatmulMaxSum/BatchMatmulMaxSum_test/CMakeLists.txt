cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(batch_matmul_max_sum_test LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "dav-2201")
endif()

# 多 case 驱动：./bmm_multi <caseId 1..15>，读 ./input/x1.bin x2.bin，写 ./output/y.bin
add_executable(bmm_multi main_multi.asc)
target_link_libraries(bmm_multi PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)
target_include_directories(bmm_multi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(bmm_multi PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>)

# 设备侧事件计时基准（best-of-10，口径对齐 OJ TBest）：./bmm_bench <caseId 1..15>
add_executable(bmm_bench bench_case.asc)
target_link_libraries(bmm_bench PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)
target_include_directories(bmm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(bmm_bench PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>)
