# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/cann_camp_2024/SinhCustomSample/SinhCustom

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out

# Utility rule file for ops_info_gen_ascend910.

# Include any custom commands dependencies for this target.
include op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/compiler_depend.make

# Include the progress variables for this target.
include op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/progress.make

op_kernel/CMakeFiles/ops_info_gen_ascend910: op_kernel/tbe/op_info_cfg/ai_core/ascend910/aic-ascend910-ops-info.json

op_kernel/tbe/op_info_cfg/ai_core/ascend910/aic-ascend910-ops-info.json:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating tbe/op_info_cfg/ai_core/ascend910/aic-ascend910-ops-info.json"
	cd /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/op_kernel && mkdir -p /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/op_kernel/tbe/op_info_cfg/ai_core/ascend910
	cd /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/op_kernel && python3 /root/cann_camp_2024/SinhCustomSample/SinhCustom/cmake/util/parse_ini_to_json.py /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/autogen/aic-ascend910-ops-info.ini /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/op_kernel/tbe/op_info_cfg/ai_core/ascend910/aic-ascend910-ops-info.json

ops_info_gen_ascend910: op_kernel/CMakeFiles/ops_info_gen_ascend910
ops_info_gen_ascend910: op_kernel/tbe/op_info_cfg/ai_core/ascend910/aic-ascend910-ops-info.json
ops_info_gen_ascend910: op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/build.make
.PHONY : ops_info_gen_ascend910

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/build: ops_info_gen_ascend910
.PHONY : op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/build

op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/clean:
	cd /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/ops_info_gen_ascend910.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/clean

op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/depend:
	cd /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/cann_camp_2024/SinhCustomSample/SinhCustom /root/cann_camp_2024/SinhCustomSample/SinhCustom/op_kernel /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/op_kernel /root/cann_camp_2024/SinhCustomSample/SinhCustom/build_out/op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_kernel/CMakeFiles/ops_info_gen_ascend910.dir/depend

