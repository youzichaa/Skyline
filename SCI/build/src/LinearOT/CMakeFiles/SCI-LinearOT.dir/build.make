# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chenyu97/github.com/EzPC/SCI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chenyu97/github.com/EzPC/SCI/build

# Include any dependencies generated for this target.
include src/LinearOT/CMakeFiles/SCI-LinearOT.dir/depend.make

# Include the progress variables for this target.
include src/LinearOT/CMakeFiles/SCI-LinearOT.dir/progress.make

# Include the compile flags for this target's objects.
include src/LinearOT/CMakeFiles/SCI-LinearOT.dir/flags.make

src/LinearOT/CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.o: src/LinearOT/CMakeFiles/SCI-LinearOT.dir/flags.make
src/LinearOT/CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.o: ../src/LinearOT/linear-ot.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenyu97/github.com/EzPC/SCI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/LinearOT/CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.o"
	cd /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.o -c /home/chenyu97/github.com/EzPC/SCI/src/LinearOT/linear-ot.cpp

src/LinearOT/CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.i"
	cd /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenyu97/github.com/EzPC/SCI/src/LinearOT/linear-ot.cpp > CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.i

src/LinearOT/CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.s"
	cd /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenyu97/github.com/EzPC/SCI/src/LinearOT/linear-ot.cpp -o CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.s

# Object files for target SCI-LinearOT
SCI__LinearOT_OBJECTS = \
"CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.o"

# External object files for target SCI-LinearOT
SCI__LinearOT_EXTERNAL_OBJECTS =

lib/libSCI-LinearOT.a: src/LinearOT/CMakeFiles/SCI-LinearOT.dir/linear-ot.cpp.o
lib/libSCI-LinearOT.a: src/LinearOT/CMakeFiles/SCI-LinearOT.dir/build.make
lib/libSCI-LinearOT.a: src/LinearOT/CMakeFiles/SCI-LinearOT.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chenyu97/github.com/EzPC/SCI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../lib/libSCI-LinearOT.a"
	cd /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT && $(CMAKE_COMMAND) -P CMakeFiles/SCI-LinearOT.dir/cmake_clean_target.cmake
	cd /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SCI-LinearOT.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/LinearOT/CMakeFiles/SCI-LinearOT.dir/build: lib/libSCI-LinearOT.a

.PHONY : src/LinearOT/CMakeFiles/SCI-LinearOT.dir/build

src/LinearOT/CMakeFiles/SCI-LinearOT.dir/clean:
	cd /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT && $(CMAKE_COMMAND) -P CMakeFiles/SCI-LinearOT.dir/cmake_clean.cmake
.PHONY : src/LinearOT/CMakeFiles/SCI-LinearOT.dir/clean

src/LinearOT/CMakeFiles/SCI-LinearOT.dir/depend:
	cd /home/chenyu97/github.com/EzPC/SCI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenyu97/github.com/EzPC/SCI /home/chenyu97/github.com/EzPC/SCI/src/LinearOT /home/chenyu97/github.com/EzPC/SCI/build /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT /home/chenyu97/github.com/EzPC/SCI/build/src/LinearOT/CMakeFiles/SCI-LinearOT.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/LinearOT/CMakeFiles/SCI-LinearOT.dir/depend

