cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-experimental-semisupervised)

# Build a version of the wav2letter++ object containing default wav2letter++ interfaces
# minus interface libraries which were custom defined, which are linked explicitly
add_library(
  wav2letter++_ss
  ""
  )

# Add custom targets
add_subdirectory(${PROJECT_SOURCE_DIR}/src/module) # for target module_ss
add_subdirectory(${PROJECT_SOURCE_DIR}/src/runtime) # for target runtime_ss

target_link_libraries(
  wav2letter++_ss
  PUBLIC
  wav2letter++
  module_ss    # semisupervised-specific
  runtime_ss   # semisupervised-specific
  convlm-serializer # see src/experimental/ConvLM
  )

# ------------------------- Dedicated SSL Train executable --------------------------
add_executable(
  Train_ss
  Train.cpp
  )

target_link_libraries(
  Train_ss
  wav2letter++_ss
)
