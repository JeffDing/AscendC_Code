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

# Utility rule file for ascendc_impl_gen.

# Include any custom commands dependencies for this target.
include op_kernel/CMakeFiles/ascendc_impl_gen.dir/compiler_depend.make

# Include the progress variables for this target.
include op_kernel/CMakeFiles/ascendc_impl_gen.dir/progress.make

op_kernel/CMakeFiles/ascendc_impl_gen: op_kernel/tbe/.impl_timestamp

op_kernel/tbe/.impl_timestamp: autogen/aic-ascend910b-ops-info.ini
op_kernel/tbe/.impl_timestamp: ../cmake/util/ascendc_impl_build.py
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/tmp/code/SinhCustom/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating tbe/.impl_timestamp"
	cd /tmp/code/SinhCustom/build_out/op_kernel && mkdir -m 700 -p /tmp/code/SinhCustom/build_out/op_kernel/tbe/dynamic
	cd /tmp/code/SinhCustom/build_out/op_kernel && python3 /tmp/code/SinhCustom/cmake/util/ascendc_impl_build.py /tmp/code/SinhCustom/build_out/autogen/aic-ascend910b-ops-info.ini "" "" /tmp/code/SinhCustom/op_kernel /tmp/code/SinhCustom/build_out/op_kernel/tbe/dynamic /tmp/code/SinhCustom/build_out/autogen
	cd /tmp/code/SinhCustom/build_out/op_kernel && rm -rf /tmp/code/SinhCustom/build_out/op_kernel/tbe/.impl_timestamp
	cd /tmp/code/SinhCustom/build_out/op_kernel && touch /tmp/code/SinhCustom/build_out/op_kernel/tbe/.impl_timestamp

ascendc_impl_gen: op_kernel/CMakeFiles/ascendc_impl_gen
ascendc_impl_gen: op_kernel/tbe/.impl_timestamp
ascendc_impl_gen: op_kernel/CMakeFiles/ascendc_impl_gen.dir/build.make
.PHONY : ascendc_impl_gen

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/ascendc_impl_gen.dir/build: ascendc_impl_gen
.PHONY : op_kernel/CMakeFiles/ascendc_impl_gen.dir/build

op_kernel/CMakeFiles/ascendc_impl_gen.dir/clean:
	cd /tmp/code/SinhCustom/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/ascendc_impl_gen.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/ascendc_impl_gen.dir/clean

op_kernel/CMakeFiles/ascendc_impl_gen.dir/depend:
	cd /tmp/code/SinhCustom/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/code/SinhCustom /tmp/code/SinhCustom/op_kernel /tmp/code/SinhCustom/build_out /tmp/code/SinhCustom/build_out/op_kernel /tmp/code/SinhCustom/build_out/op_kernel/CMakeFiles/ascendc_impl_gen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_kernel/CMakeFiles/ascendc_impl_gen.dir/depend
