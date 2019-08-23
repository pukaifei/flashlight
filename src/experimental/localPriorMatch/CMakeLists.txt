cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-experimental-localPriorMatch)

# Build a version of the wav2letter++ object containing default wav2letter++ interfaces
# minus interface libraries which were custom defined, which are linked explicitly
add_library(
  wav2letter++_lpm
  ""
  )

# Add custom targets
add_subdirectory(${PROJECT_SOURCE_DIR}/src/module) # for target module_lpm
add_subdirectory(${PROJECT_SOURCE_DIR}/src/runtime) # for target runtime_lpm

target_link_libraries(
  wav2letter++_lpm
  PUBLIC
  wav2letter++
  runtime_lpm         # localPriorMatch-specific
  module_lpm          # localPriorMatch-specific
  convlm-experimental # see src/experimental/ConvLM
  )

# ------- Local Prior Matching (Tied Proposal Model) -------
add_executable(
  Train_lpm_tied
  Train_lpm_tied.cpp
)

target_link_libraries(
  Train_lpm_tied
  wav2letter++_lpm
)

# ------- Local Prior Matching (Off-Policy Proposal Model) -------
add_executable(
  Train_lpm
  Train_lpm.cpp
)

target_link_libraries(
  Train_lpm
  wav2letter++_lpm
)
 
