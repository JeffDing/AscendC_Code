# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /tmp/code/cmake-3.20.0-rc5-linux-aarch64/bin/cmake

# The command to remove a file.
RM = /tmp/code/cmake-3.20.0-rc5-linux-aarch64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /tmp/code/SinhCustom

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/code/SinhCustom/build_out

# Include any dependencies generated for this target.
include op_host/CMakeFiles/cust_opapi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include op_host/CMakeFiles/cust_opapi.dir/compiler_depend.make

# Include the progress variables for this target.
include op_host/CMakeFiles/cust_opapi.dir/progress.make

# Include the compile flags for this target's objects.
include op_host/CMakeFiles/cust_opapi.dir/flags.make

op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o: op_host/CMakeFiles/cust_opapi.dir/flags.make
op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o: autogen/aclnn_sinh_custom.cpp
op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o: op_host/CMakeFiles/cust_opapi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/code/SinhCustom/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o"
	cd /tmp/code/SinhCustom/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o -MF CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o.d -o CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o -c /tmp/code/SinhCustom/build_out/autogen/aclnn_sinh_custom.cpp

op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.i"
	cd /tmp/code/SinhCustom/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/code/SinhCustom/build_out/autogen/aclnn_sinh_custom.cpp > CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.i

op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.s"
	cd /tmp/code/SinhCustom/build_out/op_host && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/code/SinhCustom/build_out/autogen/aclnn_sinh_custom.cpp -o CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.s

# Object files for target cust_opapi
cust_opapi_OBJECTS = \
"CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o"

# External object files for target cust_opapi
cust_opapi_EXTERNAL_OBJECTS =

op_host/libcust_opapi.so: op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o
op_host/libcust_opapi.so: op_host/CMakeFiles/cust_opapi.dir/build.make
op_host/libcust_opapi.so: op_host/CMakeFiles/cust_opapi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/code/SinhCustom/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcust_opapi.so"
	cd /tmp/code/SinhCustom/build_out/op_host && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_opapi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
op_host/CMakeFiles/cust_opapi.dir/build: op_host/libcust_opapi.so
.PHONY : op_host/CMakeFiles/cust_opapi.dir/build

op_host/CMakeFiles/cust_opapi.dir/clean:
	cd /tmp/code/SinhCustom/build_out/op_host && $(CMAKE_COMMAND) -P CMakeFiles/cust_opapi.dir/cmake_clean.cmake
.PHONY : op_host/CMakeFiles/cust_opapi.dir/clean

op_host/CMakeFiles/cust_opapi.dir/depend:
	cd /tmp/code/SinhCustom/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/code/SinhCustom /tmp/code/SinhCustom/op_host /tmp/code/SinhCustom/build_out /tmp/code/SinhCustom/build_out/op_host /tmp/code/SinhCustom/build_out/op_host/CMakeFiles/cust_opapi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_host/CMakeFiles/cust_opapi.dir/depend

