cmake_minimum_required(VERSION 3.5.1)

# Experimental projects to build.
# To add your experimental project:
# - Add a subdirectory in `src/experimental`
# - Add a CMakeLists.txt file in that directory which adds your source files
#   as target_sources to the flashlight target, as well as link any other
#   libraries or include directories that need to be added for example:
#
#   include(${CMAKE_CURRENT_LIST_DIR}/[some_project_dir]/CMakeLists.txt)
#
# Add includes here:
include(${CMAKE_CURRENT_LIST_DIR}/memory/CMakeLists.txt)

if (FL_BUILD_TESTS)
  include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
endif()
