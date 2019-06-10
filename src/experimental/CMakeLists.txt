cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-experimental)

# Experimental projects to build.
# To add your experimental project:
# - Add a subdirectory in `src/experimental`
# - Add a CMakeLists.txt file in that subdirectory which builds your targets
#   (this target can link to other dependencies, e.g. flashlight::flashlight)
# - Add a call to add_subdirectory below for your experimental projects directory
add_subdirectory(${PROJECT_SOURCE_DIR}/ConvLM)
add_subdirectory(${PROJECT_SOURCE_DIR}/semisupervised)
