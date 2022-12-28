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
include networks/CMakeFiles/resnet32_cifar-HE.dir/depend.make

# Include the progress variables for this target.
include networks/CMakeFiles/resnet32_cifar-HE.dir/progress.make

# Include the compile flags for this target's objects.
include networks/CMakeFiles/resnet32_cifar-HE.dir/flags.make

networks/CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.o: networks/CMakeFiles/resnet32_cifar-HE.dir/flags.make
networks/CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.o: ../networks/main_resnet32_cifar.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenyu97/github.com/EzPC/SCI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object networks/CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.o"
	cd /home/chenyu97/github.com/EzPC/SCI/build/networks && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.o -c /home/chenyu97/github.com/EzPC/SCI/networks/main_resnet32_cifar.cpp

networks/CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.i"
	cd /home/chenyu97/github.com/EzPC/SCI/build/networks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenyu97/github.com/EzPC/SCI/networks/main_resnet32_cifar.cpp > CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.i

networks/CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.s"
	cd /home/chenyu97/github.com/EzPC/SCI/build/networks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenyu97/github.com/EzPC/SCI/networks/main_resnet32_cifar.cpp -o CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.s

# Object files for target resnet32_cifar-HE
resnet32_cifar__HE_OBJECTS = \
"CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.o"

# External object files for target resnet32_cifar-HE
resnet32_cifar__HE_EXTERNAL_OBJECTS =

bin/resnet32_cifar-HE: networks/CMakeFiles/resnet32_cifar-HE.dir/main_resnet32_cifar.cpp.o
bin/resnet32_cifar-HE: networks/CMakeFiles/resnet32_cifar-HE.dir/build.make
bin/resnet32_cifar-HE: lib/libSCI-HE.a
bin/resnet32_cifar-HE: lib/libSCI-LinearHE.a
bin/resnet32_cifar-HE: /usr/lib/x86_64-linux-gnu/libssl.so
bin/resnet32_cifar-HE: /usr/lib/x86_64-linux-gnu/libcrypto.so
bin/resnet32_cifar-HE: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/resnet32_cifar-HE: lib/libseal.a
bin/resnet32_cifar-HE: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
bin/resnet32_cifar-HE: /usr/lib/x86_64-linux-gnu/libpthread.so
bin/resnet32_cifar-HE: networks/CMakeFiles/resnet32_cifar-HE.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chenyu97/github.com/EzPC/SCI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/resnet32_cifar-HE"
	cd /home/chenyu97/github.com/EzPC/SCI/build/networks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/resnet32_cifar-HE.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
networks/CMakeFiles/resnet32_cifar-HE.dir/build: bin/resnet32_cifar-HE

.PHONY : networks/CMakeFiles/resnet32_cifar-HE.dir/build

networks/CMakeFiles/resnet32_cifar-HE.dir/clean:
	cd /home/chenyu97/github.com/EzPC/SCI/build/networks && $(CMAKE_COMMAND) -P CMakeFiles/resnet32_cifar-HE.dir/cmake_clean.cmake
.PHONY : networks/CMakeFiles/resnet32_cifar-HE.dir/clean

networks/CMakeFiles/resnet32_cifar-HE.dir/depend:
	cd /home/chenyu97/github.com/EzPC/SCI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenyu97/github.com/EzPC/SCI /home/chenyu97/github.com/EzPC/SCI/networks /home/chenyu97/github.com/EzPC/SCI/build /home/chenyu97/github.com/EzPC/SCI/build/networks /home/chenyu97/github.com/EzPC/SCI/build/networks/CMakeFiles/resnet32_cifar-HE.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : networks/CMakeFiles/resnet32_cifar-HE.dir/depend

