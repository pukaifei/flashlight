cmake_minimum_required(VERSION 3.5.1)

add_library(
  module_lpm_oss
  INTERFACE
  )

target_sources(
  module_lpm_oss
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/LMCritic.cpp
  )

target_link_libraries(
  module_lpm_oss
  INTERFACE
  module # inherit all properties from base interface
  )
