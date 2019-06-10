cmake_minimum_required(VERSION 3.5.1)

add_library(
  module_ss
  INTERFACE
  )

target_sources(
  module_ss
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/LMCritic.cpp
  )

target_link_libraries(
  module_ss
  INTERFACE
  module # inherit all properties from base interface
  )
