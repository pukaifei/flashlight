cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-experimental-localPriorMatchOss)

# Build a version of the wav2letter++ object containing default wav2letter++ interfaces
# minus interface libraries which were custom defined, which are linked explicitly
add_library(
  wav2letter++_lpm_oss
  ""
  )

# Add custom targets
add_subdirectory(${PROJECT_SOURCE_DIR}/src/module) # for target module_lpm_oss
add_subdirectory(${PROJECT_SOURCE_DIR}/src/runtime) # for target runtime_lpm_oss

target_link_libraries(
  wav2letter++_lpm_oss
  PUBLIC
  wav2letter++
  runtime_lpm_oss         # localPriorMatchOss-specific
  module_lpm_oss          # localPriorMatchOss-specific
  convlm-experimental # see src/experimental/ConvLM
  )

# ------- Local Prior Matching (Tied Proposal Model) -------
add_executable(
  Train_lpm_tied_oss
  Train_lpm_tied.cpp
)

target_link_libraries(
  Train_lpm_tied_oss
  wav2letter++_lpm_oss
)

# ------- Local Prior Matching (Off-Policy Proposal Model) -------
add_executable(
  Train_lpm_oss
  Train_lpm.cpp
)

target_link_libraries(
  Train_lpm_oss
  wav2letter++_lpm_oss
)
 
