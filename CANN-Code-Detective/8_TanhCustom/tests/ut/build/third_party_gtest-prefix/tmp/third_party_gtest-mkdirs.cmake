# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/code/tests/ut/build/3rd_party/gtest-src"
  "/workspace/code/tests/ut/build/third_party_gtest-prefix/src/third_party_gtest-build"
  "/workspace/code/tests/ut/build/3rd_party/gtest"
  "/workspace/code/tests/ut/build/third_party_gtest-prefix/tmp"
  "/workspace/code/tests/ut/build/third_party_gtest-prefix/src/third_party_gtest-stamp"
  "/workspace/code/tests/ut/build/downloads"
  "/workspace/code/tests/ut/build/third_party_gtest-prefix/src/third_party_gtest-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/code/tests/ut/build/third_party_gtest-prefix/src/third_party_gtest-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/code/tests/ut/build/third_party_gtest-prefix/src/third_party_gtest-stamp${cfgdir}") # cfgdir has leading slash
endif()
